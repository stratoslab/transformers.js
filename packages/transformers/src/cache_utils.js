import { Tensor } from './utils/tensor.js';

/**
 * Extract a dense tensor map from decoder results.
 * @param {Object} decoderResults The decoder results object.
 * @param {Record<string, Tensor>|null} [pastKeyValues=null] Previous dense tensor map, if available.
 * @param {boolean} [disposeEncoderPKVs=false] Whether to dispose encoder past key values.
 * @returns {Record<string, Tensor>} A name-to-tensor mapping for all cache outputs.
 */
export function buildPastKeyValuesTensorMap(decoderResults, pastKeyValues = null, disposeEncoderPKVs = false) {
    /** @type {Record<string, Tensor>} */
    const pkvs = Object.create(null);

    for (const name in decoderResults) {
        if (!name.startsWith('present')) continue;

        const newName = name
            // Hybrid cache architecture
            .replace('present_ssm', 'past_ssm') // Mamba
            .replace('present_conv', 'past_conv') // LFM2
            .replace('present_recurrent', 'past_recurrent') // Qwen3.5

            // Standard cache architecture
            .replace('present', 'past_key_values');

        const is_encoder_pkv = name.includes('encoder');
        if (is_encoder_pkv && pastKeyValues) {
            pkvs[newName] = pastKeyValues[newName];
        } else {
            pkvs[newName] = decoderResults[name];
        }

        if (pastKeyValues && (!is_encoder_pkv || disposeEncoderPKVs)) {
            const t = pastKeyValues[newName];
            if (t?.location === 'gpu-buffer') {
                t.dispose();
            }
        }
    }

    return pkvs;
}

function getPastKeyValuesName(name) {
    return name
        .replace('present_ssm', 'past_ssm')
        .replace('present_conv', 'past_conv')
        .replace('present_recurrent', 'past_recurrent')
        .replace('present', 'past_key_values');
}

function cloneTensorData(data) {
    return typeof data.slice === 'function' ? data.slice() : Array.from(data);
}

function packBits(codes, bits) {
    const words = new Uint32Array(Math.ceil((codes.length * bits) / 32));
    const mask = (1 << bits) - 1;
    let bitOffset = 0;

    for (let i = 0; i < codes.length; ++i) {
        const code = codes[i] & mask;
        const wordIndex = bitOffset >>> 5;
        const shift = bitOffset & 31;
        words[wordIndex] |= code << shift;

        const spill = shift + bits - 32;
        if (spill > 0) {
            words[wordIndex + 1] |= code >>> (bits - spill);
        }
        bitOffset += bits;
    }

    return words;
}

function unpackBits(words, length, bits) {
    const codes = new Uint8Array(length);
    const mask = (1 << bits) - 1;
    let bitOffset = 0;

    for (let i = 0; i < length; ++i) {
        const wordIndex = bitOffset >>> 5;
        const shift = bitOffset & 31;
        let code = (words[wordIndex] >>> shift) & mask;

        const spill = shift + bits - 32;
        if (spill > 0) {
            code |= (words[wordIndex + 1] & ((1 << spill) - 1)) << (bits - spill);
        }
        codes[i] = code;
        bitOffset += bits;
    }

    return codes;
}

function isPowerOfTwo(value) {
    return value > 0 && (value & (value - 1)) === 0;
}

function createRademacherSigns(dim, seed) {
    const signs = new Float32Array(dim);
    let state = seed >>> 0;
    for (let i = 0; i < dim; ++i) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        signs[i] = state & 1 ? 1 : -1;
    }
    return signs;
}

function hadamardInPlace(buffer) {
    for (let len = 1; len < buffer.length; len <<= 1) {
        const step = len << 1;
        for (let i = 0; i < buffer.length; i += step) {
            for (let j = 0; j < len; ++j) {
                const a = buffer[i + j];
                const b = buffer[i + j + len];
                buffer[i + j] = a + b;
                buffer[i + j + len] = a - b;
            }
        }
    }
}

function applyRotation(data, dims, seed) {
    const headDim = dims.at(-1) ?? 0;
    if (!isPowerOfTwo(headDim) || headDim === 0) {
        return { data, rotated: false, seed: null };
    }

    const vectors = data.length / headDim;
    const signs = createRademacherSigns(headDim, seed);
    const rotated = new Float32Array(data.length);
    const scratch = new Float32Array(headDim);
    const norm = 1 / Math.sqrt(headDim);

    for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
        const offset = vectorIndex * headDim;
        for (let i = 0; i < headDim; ++i) {
            scratch[i] = data[offset + i] * signs[i];
        }
        hadamardInPlace(scratch);
        for (let i = 0; i < headDim; ++i) {
            rotated[offset + i] = scratch[i] * norm;
        }
    }

    return { data: rotated, rotated: true, seed };
}

function invertRotation(data, dims, seed) {
    const headDim = dims.at(-1) ?? 0;
    if (!isPowerOfTwo(headDim) || headDim === 0) {
        return data;
    }

    const vectors = data.length / headDim;
    const signs = createRademacherSigns(headDim, seed);
    const restored = new Float32Array(data.length);
    const scratch = new Float32Array(headDim);
    const norm = 1 / Math.sqrt(headDim);

    for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
        const offset = vectorIndex * headDim;
        for (let i = 0; i < headDim; ++i) {
            scratch[i] = data[offset + i];
        }
        hadamardInPlace(scratch);
        for (let i = 0; i < headDim; ++i) {
            restored[offset + i] = scratch[i] * norm * signs[i];
        }
    }

    return restored;
}

function quantizeArray(data, bits) {
    const length = data.length;
    if (length === 0) {
        return {
            bits,
            min: 0,
            scale: 1,
            length: 0,
            packed: new Uint32Array(0),
        };
    }

    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < length; ++i) {
        const value = data[i];
        if (value < min) min = value;
        if (value > max) max = value;
    }

    const levels = (1 << bits) - 1;
    const range = max - min;
    const scale = range > 0 ? range / levels : 1;
    const codes = new Uint8Array(length);

    if (range > 0) {
        for (let i = 0; i < length; ++i) {
            const normalized = Math.round((data[i] - min) / scale);
            codes[i] = Math.max(0, Math.min(levels, normalized));
        }
    }

    return {
        bits,
        min,
        scale,
        length,
        packed: packBits(codes, bits),
    };
}

function dequantizeArray(quantized) {
    const codes = unpackBits(quantized.packed, quantized.length, quantized.bits);
    const restored = new Float32Array(quantized.length);
    for (let i = 0; i < codes.length; ++i) {
        restored[i] = quantized.min + quantized.scale * codes[i];
    }
    return restored;
}

function packTurboQuantTensor(tensor, bits, { seed = 1337, residualBits = 1, residual = false } = {}) {
    const source = tensor.type === 'float32' ? tensor : tensor.to('float32');
    const rotated = applyRotation(source.data, source.dims, seed);
    const quantized = quantizeArray(rotated.data, bits);

    /** @type {Float32Array|null} */
    let residualNorms = null;
    /** @type {Uint32Array|null} */
    let residualPacked = null;

    if (residual && residualBits === 1 && (source.dims.at(-1) ?? 0) > 0 && quantized.length > 0) {
        const approx = dequantizeArray(quantized);
        const headDim = source.dims.at(-1);
        const vectors = approx.length / headDim;
        residualNorms = new Float32Array(vectors);
        const residualCodes = new Uint8Array(approx.length);

        for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
            const offset = vectorIndex * headDim;
            let normSq = 0;
            for (let i = 0; i < headDim; ++i) {
                const value = rotated.data[offset + i] - approx[offset + i];
                residualCodes[offset + i] = value >= 0 ? 1 : 0;
                normSq += value * value;
            }
            residualNorms[vectorIndex] = Math.sqrt(normSq);
        }

        residualPacked = packBits(residualCodes, residualBits);
    }

    return {
        format: 'turboquant',
        originalType: tensor.type,
        dims: tensor.dims.slice(),
        rotationSeed: rotated.seed,
        rotated: rotated.rotated,
        quantized,
        residualBits,
        residualPacked,
        residualNorms,
    };
}

function unpackQuantizedTensor(packed) {
    if (packed.format === 'turboquant') {
        const approx = dequantizeArray(packed.quantized);
        let corrected = approx;

        if (packed.residualPacked && packed.residualNorms) {
            const headDim = packed.dims.at(-1);
            const vectors = approx.length / headDim;
            const signs = unpackBits(packed.residualPacked, approx.length, packed.residualBits);
            corrected = new Float32Array(approx.length);
            corrected.set(approx);

            const scaleBase = 1 / Math.sqrt(headDim);
            for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
                const offset = vectorIndex * headDim;
                const amplitude = packed.residualNorms[vectorIndex] * scaleBase;
                for (let i = 0; i < headDim; ++i) {
                    corrected[offset + i] += signs[offset + i] ? amplitude : -amplitude;
                }
            }
        }

        const restored = packed.rotated ? invertRotation(corrected, packed.dims, packed.rotationSeed) : corrected;
        let tensor = new Tensor('float32', restored, packed.dims.slice());
        if (packed.originalType !== 'float32') {
            tensor = tensor.to(packed.originalType);
        }
        return tensor;
    }

    const restored = dequantizeArray(packed);

    let tensor = new Tensor('float32', restored, packed.dims.slice());
    if (packed.originalType !== 'float32') {
        tensor = tensor.to(packed.originalType);
    }
    return tensor;
}

function packDenseTensor(tensor) {
    return {
        format: 'dense',
        type: tensor.type,
        dims: tensor.dims.slice(),
        data: cloneTensorData(tensor.data),
    };
}

function unpackDenseTensor(packed) {
    if (packed.format === 'quantized' || packed.format === 'turboquant') {
        return unpackQuantizedTensor(packed);
    }
    return new Tensor(packed.type, cloneTensorData(packed.data), packed.dims.slice());
}

function getTypedArrayByteLength(data) {
    return data?.byteLength ?? 0;
}

function getPackedEntrySize(packed) {
    if (packed.format === 'dense') {
        return getTypedArrayByteLength(packed.data);
    }

    if (packed.format === 'quantized') {
        return getTypedArrayByteLength(packed.packed) + 8;
    }

    if (packed.format === 'turboquant') {
        return (
            getTypedArrayByteLength(packed.quantized.packed) +
            getTypedArrayByteLength(packed.residualPacked) +
            getTypedArrayByteLength(packed.residualNorms) +
            16
        );
    }

    return 0;
}

/**
 * Base class for generation caches.
 *
 * Custom cache implementations can override:
 * - `update(decoderResults, options)` to ingest decoder outputs.
 * - `materialize(decoderFeeds)` to provide dense tensors before a decoder call.
 *
 * `materialize()` may return either a plain tensor map or `{ entries, cleanup }`.
 * If `cleanup` is returned, it will be awaited after the decoder session completes.
 */
class _PastKeyValues {
    owns_decoder_results = false;

    /**
     * Get the cached sequence length.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        throw new Error('PastKeyValues.get_seq_length() is not implemented.');
    }

    /**
     * Update the cache from decoder outputs.
     * @param {Object} decoderResults Decoder outputs.
     * @param {{ disposeEncoderPKVs?: boolean, disposeSourceDecoderResults?: boolean }} [options]
     * @returns {PastKeyValues}
     */
    update(decoderResults, options = {}) {
        return new DynamicCache(buildPastKeyValuesTensorMap(decoderResults, null, options.disposeEncoderPKVs ?? false));
    }

    /**
     * Materialize dense cache tensors for the next decoder call.
     * @returns {Record<string, Tensor>|{ entries: Record<string, Tensor>, cleanup?: (() => void|Promise<void>) }}}
     */
    materialize() {
        throw new Error('PastKeyValues.materialize() is not implemented.');
    }

    /**
     * Dispose any backing resources.
     * @returns {Promise<void>}
     */
    async dispose() {}

    /**
     * Return implementation-specific cache statistics.
     * @returns {{implementation: string, entries: number, seq_length: number, packed_bytes: number, dense_bytes: number}}
     */
    getStats() {
        return {
            implementation: 'custom',
            entries: 0,
            seq_length: this.get_seq_length?.() ?? 0,
            packed_bytes: 0,
            dense_bytes: 0,
        };
    }
}

/**
 * A cache class that stores past key values as named tensors.
 */
class _DynamicCache extends _PastKeyValues {
    /**
     * Create a DynamicCache, optionally pre-populated with entries.
     * @param {Record<string, Tensor>} [entries] Initial name→Tensor mappings.
     */
    constructor(entries) {
        if (!entries) return;
        for (const key in entries) {
            if (key in this) {
                throw new TypeError(`Key "${key}" conflicts with an existing property on DynamicCache`);
            }
            const value = entries[key];
            if (!(value instanceof Tensor)) {
                throw new TypeError(`Expected a Tensor for key "${key}", got ${typeof value}`);
            }
            this[key] = value;
        }
    }

    /**
     * Get the cached sequence length. This requires at least one attention cache entry to be present.
     * @returns {number} The past sequence length.
     */
    get_seq_length() {
        /** @type {Record<string, Tensor>} */
        const self = /** @type {any} */ (this);
        for (const name in self) {
            if (name.startsWith('past_key_values.')) {
                return self[name].dims.at(-2);
            }
        }
        throw new Error('Unable to determine sequence length from the cache.');
    }

    /**
     * Update the dense cache from decoder outputs.
     * @param {Object} decoderResults Decoder outputs.
     * @param {{ disposeEncoderPKVs?: boolean, disposeSourceDecoderResults?: boolean }} [options]
     * @returns {DynamicCache}
     */
    update(decoderResults, options = {}) {
        return new DynamicCache(
            buildPastKeyValuesTensorMap(decoderResults, this, options.disposeEncoderPKVs ?? false),
        );
    }

    /**
     * Return the cache as a dense tensor map.
     * @returns {Record<string, Tensor>}
     */
    materialize() {
        return /** @type {Record<string, Tensor>} */ (this);
    }

    /**
     * Dispose all contained tensors whose data resides on the GPU.
     * Returns a promise that resolves when all disposals are complete.
     * @returns {Promise<void>} Promise that resolves when all GPU tensors are disposed.
     */
    async dispose() {
        const promises = [];
        for (const t of /** @type {Tensor[]} */ (Object.values(this))) {
            if (t.location === 'gpu-buffer') {
                promises.push(t.dispose());
            }
        }
        await Promise.all(promises);
    }

    getStats() {
        let denseBytes = 0;
        let entries = 0;
        for (const value of Object.values(this)) {
            if (value instanceof Tensor) {
                denseBytes += getTypedArrayByteLength(value.data);
                entries += 1;
            }
        }
        return {
            implementation: 'dynamic',
            entries,
            seq_length: entries > 0 ? this.get_seq_length() : 0,
            packed_bytes: denseBytes,
            dense_bytes: denseBytes,
        };
    }
}

/**
 * @typedef {_PastKeyValues} PastKeyValues
 */

/**
 * @typedef {_DynamicCache & Record<string, Tensor>} DynamicCache
 */

export const PastKeyValues = /** @type {new () => PastKeyValues} */ (
    /** @type {unknown} */ (_PastKeyValues)
);

export const DynamicCache = /** @type {new (entries?: Record<string, Tensor>) => DynamicCache} */ (
    /** @type {unknown} */ (_DynamicCache)
);

/**
 * Experimental cache scaffold for TurboQuant-style KV compression.
 *
 * This class currently owns the lifecycle needed for custom cache implementations:
 * - packs decoder `present.*` tensors into CPU-backed buffers on update
 * - reconstructs dense `past_key_values.*` tensors before the next decoder call
 *
 * The current packing format is intentionally conservative and stores dense values.
 * Future work will replace `packDenseTensor` / `unpackDenseTensor` with a true
 * PolarQuant + QJL representation while preserving the same cache contract.
 */
class _TurboQuantCache extends _PastKeyValues {
    owns_decoder_results = true;

    constructor(config = {}) {
        super();
        this.config = {
            b_key: 3,
            b_value: 3,
            residual_length: 128,
            ...config,
        };
        this.entries = Object.create(null);
        this.seq_length = 0;
    }

    get_seq_length() {
        return this.seq_length;
    }

    /**
     * Pack decoder outputs into CPU-backed buffers.
     * @param {Object} decoderResults Decoder outputs.
     * @param {{ disposeSourceDecoderResults?: boolean }} [options]
     * @returns {TurboQuantCache}
     */
    update(decoderResults, options = {}) {
        const next = new TurboQuantCache(this.config);

        for (const name in decoderResults) {
            if (!name.startsWith('present')) continue;

            const pastName = getPastKeyValuesName(name);
            const tensor = decoderResults[name];
            const bits = pastName.endsWith('.key') ? this.config.b_key : this.config.b_value;
            next.entries[pastName] =
                bits >= 8
                    ? packDenseTensor(tensor)
                    : packTurboQuantTensor(tensor, bits, {
                          seed: this.config.rotation_seed ?? 1337,
                          residualBits: 1,
                          residual: pastName.endsWith('.key'),
                      });

            if (name.startsWith('present.') && tensor.dims.length >= 3) {
                next.seq_length = tensor.dims.at(-2);
            }

            if (options.disposeSourceDecoderResults && tensor.location === 'gpu-buffer') {
                tensor.dispose();
            }
        }

        return next;
    }

    /**
     * Rebuild dense tensors for ONNX Runtime feeds.
     * @returns {Record<string, Tensor>}
     */
    materialize() {
        /** @type {Record<string, Tensor>} */
        const entries = Object.create(null);
        for (const name in this.entries) {
            entries[name] = unpackDenseTensor(this.entries[name]);
        }
        return entries;
    }

    async dispose() {
        this.entries = Object.create(null);
        this.seq_length = 0;
    }

    getStats() {
        let packedBytes = 0;
        let denseBytes = 0;
        let entries = 0;

        for (const packed of Object.values(this.entries)) {
            packedBytes += getPackedEntrySize(packed);
            const dims = packed.dims ?? [];
            const size = dims.reduce((a, b) => a * b, 1);
            const bytesPerElement =
                packed.originalType === 'float16' || packed.type === 'float16'
                    ? 2
                    : packed.originalType === 'float32' || packed.type === 'float32'
                      ? 4
                      : 4;
            denseBytes += size * bytesPerElement;
            entries += 1;
        }

        return {
            implementation: 'turboquant',
            entries,
            seq_length: this.seq_length,
            packed_bytes: packedBytes,
            dense_bytes: denseBytes,
        };
    }
}

/**
 * @typedef {_TurboQuantCache} TurboQuantCache
 */

export const TurboQuantCache = /** @type {new (config?: Object) => TurboQuantCache} */ (
    /** @type {unknown} */ (_TurboQuantCache)
);
