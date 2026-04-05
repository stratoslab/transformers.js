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

function packQuantizedTensor(tensor, bits) {
    const source = tensor.type === 'float32' ? tensor : tensor.to('float32');
    const data = source.data;
    const length = data.length;

    if (length === 0) {
        return {
            format: 'quantized',
            originalType: tensor.type,
            dims: tensor.dims.slice(),
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
        format: 'quantized',
        originalType: tensor.type,
        dims: tensor.dims.slice(),
        bits,
        min,
        scale,
        length,
        packed: packBits(codes, bits),
    };
}

function unpackQuantizedTensor(packed) {
    const codes = unpackBits(packed.packed, packed.length, packed.bits);
    const restored = new Float32Array(packed.length);
    for (let i = 0; i < codes.length; ++i) {
        restored[i] = packed.min + packed.scale * codes[i];
    }

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
    if (packed.format === 'quantized') {
        return unpackQuantizedTensor(packed);
    }
    return new Tensor(packed.type, cloneTensorData(packed.data), packed.dims.slice());
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
            next.entries[pastName] = bits >= 8 ? packDenseTensor(tensor) : packQuantizedTensor(tensor, bits);

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
}

/**
 * @typedef {_TurboQuantCache} TurboQuantCache
 */

export const TurboQuantCache = /** @type {new (config?: Object) => TurboQuantCache} */ (
    /** @type {unknown} */ (_TurboQuantCache)
);
