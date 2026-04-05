import { Tensor, cat } from './utils/tensor.js';
import { DataTypeMap } from './utils/dtypes.js';

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

function getBytesPerElement(type) {
    const ctor = DataTypeMap[type];
    return ctor?.BYTES_PER_ELEMENT ?? 0;
}

function getTensorByteLength(tensor) {
    return tensor.size * getBytesPerElement(tensor.type);
}

async function cloneTensorToCPU(tensor) {
    if (tensor.location === 'gpu-buffer' && typeof tensor.ort_tensor?.getData === 'function') {
        const data = await tensor.ort_tensor.getData();
        return new Tensor(tensor.type, cloneTensorData(data), tensor.dims.slice());
    }
    return new Tensor(tensor.type, cloneTensorData(tensor.data), tensor.dims.slice());
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

// Module-level memoization: Rademacher signs for each (dim, seed) pair are
// pure functions of inputs and reused across every pack/unpack call. Caching
// them avoids recomputing the LCG on every cache update (typically thousands
// of calls per generation).
const RADEMACHER_CACHE = new Map();

function createRademacherSigns(dim, seed) {
    const key = (seed >>> 0) * 0x100000000 + dim; // unique composite key
    const hit = RADEMACHER_CACHE.get(key);
    if (hit) return hit;
    const signs = new Float32Array(dim);
    let state = seed >>> 0;
    for (let i = 0; i < dim; ++i) {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        signs[i] = state & 1 ? 1 : -1;
    }
    RADEMACHER_CACHE.set(key, signs);
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

// Rotation writes directly into the output buffer: we copy the (signed) input
// into `output[offset..offset+headDim)` and run the Hadamard transform in-place
// on that slice, then scale by norm. Saves one Float32Array allocation + one
// full pass vs the previous (scratch buffer + rotated buffer) layout.
function applyRotation(data, dims, seed) {
    const headDim = dims.at(-1) ?? 0;
    if (!isPowerOfTwo(headDim) || headDim === 0) {
        return { data, rotated: false, seed: null };
    }

    const vectors = data.length / headDim;
    const signs = createRademacherSigns(headDim, seed);
    const rotated = new Float32Array(data.length);
    const norm = 1 / Math.sqrt(headDim);

    for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
        const offset = vectorIndex * headDim;
        // Pre-multiply by sign directly into output slice.
        for (let i = 0; i < headDim; ++i) {
            rotated[offset + i] = data[offset + i] * signs[i];
        }
        // In-place Hadamard on the slice (uses a subarray view — no alloc).
        hadamardInPlace(rotated.subarray(offset, offset + headDim));
        // Normalize in place.
        for (let i = 0; i < headDim; ++i) {
            rotated[offset + i] *= norm;
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
    const norm = 1 / Math.sqrt(headDim);

    for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
        const offset = vectorIndex * headDim;
        // Copy input into output, Hadamard in-place, then scale*sign.
        for (let i = 0; i < headDim; ++i) {
            restored[offset + i] = data[offset + i];
        }
        hadamardInPlace(restored.subarray(offset, offset + headDim));
        for (let i = 0; i < headDim; ++i) {
            restored[offset + i] = restored[offset + i] * norm * signs[i];
        }
    }

    return restored;
}

// Min-max quantizer: uses the full range of observed values. Fast and simple
// but wastes levels on outliers. Kept as a fallback and for correctness tests.
function quantizeArrayMinMax(data, bits) {
    const length = data.length;
    if (length === 0) {
        return { bits, min: 0, scale: 1, length: 0, packed: new Uint32Array(0) };
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
        const invScale = 1 / scale;
        for (let i = 0; i < length; ++i) {
            const normalized = Math.round((data[i] - min) * invScale);
            codes[i] = normalized < 0 ? 0 : normalized > levels ? levels : normalized;
        }
    }

    return { bits, min, scale, length, packed: packBits(codes, bits) };
}

// σ-clip quantizer: after random rotation the coordinates are approximately
// Gaussian around zero, so clipping to mean ± k·σ (k≈2.5) captures >98% of
// energy while giving ~2× tighter level spacing than min-max. This is the
// cheap Lloyd-Max-style improvement the paper calls for — at the same bit
// budget we lose less attention fidelity, so 3 bits/coord becomes viable.
function quantizeArraySigmaClip(data, bits, sigmaK) {
    const length = data.length;
    if (length === 0) {
        return { bits, min: 0, scale: 1, length: 0, packed: new Uint32Array(0) };
    }

    // Welford-style single-pass mean + M2 for numerical stability.
    let mean = 0;
    let m2 = 0;
    for (let i = 0; i < length; ++i) {
        const value = data[i];
        const delta = value - mean;
        mean += delta / (i + 1);
        m2 += delta * (value - mean);
    }
    const variance = length > 1 ? m2 / length : 0;
    const sigma = Math.sqrt(variance);

    const levels = (1 << bits) - 1;
    const halfRange = sigmaK * sigma;
    const min = mean - halfRange;
    const range = 2 * halfRange;
    const scale = range > 0 ? range / levels : 1;
    const codes = new Uint8Array(length);

    if (range > 0) {
        const invScale = 1 / scale;
        for (let i = 0; i < length; ++i) {
            const normalized = Math.round((data[i] - min) * invScale);
            codes[i] = normalized < 0 ? 0 : normalized > levels ? levels : normalized;
        }
    }

    return { bits, min, scale, length, packed: packBits(codes, bits) };
}

// Route based on caller preference; default to σ-clip for rotated data since
// it's strictly better at the same bit count under a Gaussian distribution.
function quantizeArray(data, bits, options = {}) {
    const { mode = 'sigma', sigmaK = 2.5 } = options;
    // Very small arrays lack the samples needed for a stable σ estimate;
    // fall through to min-max in that case.
    if (mode === 'sigma' && data.length >= 8) {
        return quantizeArraySigmaClip(data, bits, sigmaK);
    }
    return quantizeArrayMinMax(data, bits);
}

function dequantizeArray(quantized) {
    const codes = unpackBits(quantized.packed, quantized.length, quantized.bits);
    const restored = new Float32Array(quantized.length);
    for (let i = 0; i < codes.length; ++i) {
        restored[i] = quantized.min + quantized.scale * codes[i];
    }
    return restored;
}

async function packTurboQuantTensor(
    tensor,
    bits,
    { seed = 1337, residualBits = 1, residual = false, quantization = 'sigma', sigmaK = 2.5 } = {},
) {
    const cpuTensor = tensor.location === 'gpu-buffer' ? await cloneTensorToCPU(tensor) : tensor;
    const source = cpuTensor.type === 'float32' ? cpuTensor : cpuTensor.to('float32');
    const rotated = applyRotation(source.data, source.dims, seed);
    const quantized = quantizeArray(rotated.data, bits, { mode: quantization, sigmaK });

    /** @type {Float32Array|null} */
    let residualNorms = null;
    /** @type {Uint32Array|null} */
    let residualPacked = null;

    if (residual && residualBits === 1 && (source.dims.at(-1) ?? 0) > 0 && quantized.length > 0) {
        const approx = dequantizeArray(quantized);
        const headDim = source.dims.at(-1);
        const vectors = approx.length / headDim;
        residualNorms = new Float32Array(vectors);
        const residualMeans = new Float32Array(vectors);
        const residualCodes = new Uint8Array(approx.length);

        for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
            const offset = vectorIndex * headDim;
            let sum = 0;
            let normSq = 0;
            for (let i = 0; i < headDim; ++i) {
                const value = rotated.data[offset + i] - approx[offset + i];
                sum += value;
                normSq += value * value;
            }

            const mean = sum / headDim;
            residualMeans[vectorIndex] = mean;

            for (let i = 0; i < headDim; ++i) {
                const centered = rotated.data[offset + i] - approx[offset + i] - mean;
                residualCodes[offset + i] = centered >= 0 ? 1 : 0;
            }
            residualNorms[vectorIndex] = Math.sqrt(normSq);
        }

        residualPacked = packBits(residualCodes, residualBits);
        quantized.residualMeans = residualMeans;
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
        residualMeans: quantized.residualMeans ?? null,
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

            for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
                const offset = vectorIndex * headDim;
                const mean = packed.residualMeans?.[vectorIndex] ?? 0;
                const amplitude = packed.residualNorms[vectorIndex] / Math.sqrt(headDim);
                for (let i = 0; i < headDim; ++i) {
                    corrected[offset + i] += mean + (signs[offset + i] ? amplitude : -amplitude);
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

async function packDenseTensor(tensor) {
    const cpuTensor = tensor.location === 'gpu-buffer' ? await cloneTensorToCPU(tensor) : tensor;
    return {
        format: 'dense',
        type: cpuTensor.type,
        dims: cpuTensor.dims.slice(),
        data: cloneTensorData(cpuTensor.data),
    };
}

function computeLeading(dims) {
    let leading = 1;
    for (let i = 0; i < dims.length - 2; ++i) leading *= dims[i];
    return leading || 1;
}

// Extract [leading, toRow-fromRow, headDim] rows from a [leading, seqDim, headDim]
// tensor's flat data array. Used to pull the new token(s) off a full present.* tensor.
function extractSeqSlice(data, leading, seqDim, headDim, fromRow, toRow) {
    const rows = toRow - fromRow;
    if (rows <= 0) return null;
    const out = new Float32Array(leading * rows * headDim);
    const srcStride = seqDim * headDim;
    const dstStride = rows * headDim;
    for (let i = 0; i < leading; ++i) {
        out.set(
            data.subarray(i * srcStride + fromRow * headDim, i * srcStride + toRow * headDim),
            i * dstStride,
        );
    }
    return out;
}

// Append newRows rows (layout [leading, newRows, headDim]) onto tail
// (layout [leading, tailRows, headDim]). Returns a fresh buffer.
function appendRowsToTail(tail, tailRows, newData, newRows, leading, headDim) {
    const totalRows = tailRows + newRows;
    const out = new Float32Array(leading * totalRows * headDim);
    const tailStride = tailRows * headDim;
    const newStride = newRows * headDim;
    const outStride = totalRows * headDim;
    for (let i = 0; i < leading; ++i) {
        if (tailRows > 0) {
            out.set(tail.subarray(i * tailStride, (i + 1) * tailStride), i * outStride);
        }
        out.set(
            newData.subarray(i * newStride, (i + 1) * newStride),
            i * outStride + tailStride,
        );
    }
    return out;
}

// Split the front `evictRows` rows off a tail buffer, returning the evicted
// rows (layout [leading, evictRows, headDim]) and the remaining tail.
function takeOldestRows(tail, tailRows, evictRows, leading, headDim) {
    const remainingRows = tailRows - evictRows;
    const evicted = new Float32Array(leading * evictRows * headDim);
    const remaining = remainingRows > 0 ? new Float32Array(leading * remainingRows * headDim) : null;
    const tailStride = tailRows * headDim;
    const evictStride = evictRows * headDim;
    const remainingStride = remainingRows * headDim;
    for (let i = 0; i < leading; ++i) {
        evicted.set(tail.subarray(i * tailStride, i * tailStride + evictStride), i * evictStride);
        if (remaining) {
            remaining.set(
                tail.subarray(i * tailStride + evictStride, (i + 1) * tailStride),
                i * remainingStride,
            );
        }
    }
    return { evicted, remaining, remainingRows };
}

function splitArrayBySequence(data, dims, tailLength) {
    const seqDim = dims.at(-2) ?? 0;
    const headDim = dims.at(-1) ?? 0;
    const prefixLength = Math.max(seqDim - tailLength, 0);
    const suffixLength = seqDim - prefixLength;
    const leading = dims.slice(0, -2).reduce((a, b) => a * b, 1) || 1;
    const prefix = prefixLength > 0 ? new data.constructor(leading * prefixLength * headDim) : null;
    const suffix = suffixLength > 0 ? new data.constructor(leading * suffixLength * headDim) : null;

    if (prefixLength === 0) {
        suffix?.set(data);
        return { prefix, prefixLength, suffix, suffixLength };
    }

    const fullStride = seqDim * headDim;
    const prefixStride = prefixLength * headDim;
    const suffixStride = suffixLength * headDim;
    for (let i = 0; i < leading; ++i) {
        const inputOffset = i * fullStride;
        if (prefix) {
            prefix.set(data.subarray(inputOffset, inputOffset + prefixStride), i * prefixStride);
        }
        if (suffix) {
            suffix.set(
                data.subarray(inputOffset + prefixStride, inputOffset + prefixStride + suffixStride),
                i * suffixStride,
            );
        }
    }

    return { prefix, prefixLength, suffix, suffixLength };
}

async function packTensorWithResidualWindow(
    tensor,
    bits,
    {
        residualLength = 0,
        seed = 1337,
        residualBits = 1,
        residual = false,
        quantization = 'sigma',
        sigmaK = 2.5,
    } = {},
) {
    const cpuTensor = tensor.location === 'gpu-buffer' ? await cloneTensorToCPU(tensor) : tensor;
    const dims = cpuTensor.dims.slice();
    const seqDim = dims.at(-2) ?? 0;

    if (bits >= 8) {
        return await packDenseTensor(cpuTensor);
    }

    if (residualLength > 0 && dims.length >= 3 && seqDim <= residualLength) {
        return await packDenseTensor(cpuTensor);
    }

    if (residualLength <= 0 || dims.length < 3) {
        return await packTurboQuantTensor(cpuTensor, bits, {
            seed,
            residualBits,
            residual,
            quantization,
            sigmaK,
        });
    }

    const { prefix, prefixLength, suffix, suffixLength } = splitArrayBySequence(cpuTensor.data, dims, residualLength);
    const prefixDims = dims.slice();
    prefixDims[prefixDims.length - 2] = prefixLength;
    const suffixDims = dims.slice();
    suffixDims[suffixDims.length - 2] = suffixLength;

    const prefixTensor = prefix ? new Tensor(cpuTensor.type, prefix, prefixDims) : null;
    const suffixTensor = suffix ? new Tensor(cpuTensor.type, suffix, suffixDims) : null;

    return {
        format: 'hybrid',
        dims,
        originalType: cpuTensor.type,
        compressed: prefixTensor
            ? await packTurboQuantTensor(prefixTensor, bits, {
                  seed,
                  residualBits,
                  residual,
                  quantization,
                  sigmaK,
              })
            : null,
        tail: suffixTensor ? await packDenseTensor(suffixTensor) : null,
    };
}

// Dequantize + residual-correct + inverse-rotate a turboquant-packed block
// back into its natural [leading * blockRows * headDim] Float32Array layout.
// Factored out of unpackQuantizedTensor so the result can be scatter-copied
// into a bigger output buffer without allocating an intermediate Tensor.
function dequantizeRestoredBlock(block) {
    const approx = dequantizeArray(block.quantized);
    if (block.residualPacked && block.residualNorms) {
        const headDim = block.dims.at(-1);
        const vectors = approx.length / headDim;
        const signs = unpackBits(block.residualPacked, approx.length, block.residualBits);
        // approx was freshly allocated — mutate in place.
        for (let vectorIndex = 0; vectorIndex < vectors; ++vectorIndex) {
            const offset = vectorIndex * headDim;
            const mean = block.residualMeans?.[vectorIndex] ?? 0;
            const amplitude = block.residualNorms[vectorIndex] / Math.sqrt(headDim);
            for (let i = 0; i < headDim; ++i) {
                approx[offset + i] += mean + (signs[offset + i] ? amplitude : -amplitude);
            }
        }
    }
    return block.rotated ? invertRotation(approx, block.dims, block.rotationSeed) : approx;
}

// Write a packed block's dequantized rows into `out` at `rowOffset`, using
// the output's `outStride` (number of Float32s per leading index in `out`).
function writeBlockIntoBuffer(block, out, outStride, rowOffset, leading, headDim) {
    const restored = dequantizeRestoredBlock(block);
    const blockRows = block.dims.at(-2) ?? 0;
    const blockStride = blockRows * headDim;
    const byteRowOffset = rowOffset * headDim;
    for (let i = 0; i < leading; ++i) {
        out.set(
            restored.subarray(i * blockStride, (i + 1) * blockStride),
            i * outStride + byteRowOffset,
        );
    }
}

function unpackDenseTensor(packed) {
    if (packed.format === 'hybrid') {
        const tensors = [];
        if (packed.compressed) {
            tensors.push(unpackDenseTensor(packed.compressed));
        }
        if (packed.tail) {
            tensors.push(unpackDenseTensor(packed.tail));
        }
        if (tensors.length === 1) {
            return tensors[0];
        }
        return cat(tensors, -2);
    }

    if (packed.format === 'incremental-hybrid') {
        // Materialize directly into a single output buffer, avoiding cat() and
        // the N per-block allocations. Each block's dequantized rows are
        // written in-place into the correct [leading, row, headDim] slice of
        // the output, then tail rows are copied in.
        const headDim = packed.headDim;
        const leading = packed.leading;
        const totalRows = packed.compressedRows + packed.tailRows;
        const outDims = packed.dims.slice();
        outDims[outDims.length - 2] = totalRows;

        if (totalRows === 0) {
            const empty = new Tensor('float32', new Float32Array(0), outDims);
            return packed.originalType !== 'float32' ? empty.to(packed.originalType) : empty;
        }

        const out = new Float32Array(leading * totalRows * headDim);
        const outStride = totalRows * headDim;

        // Fill compressed blocks in order.
        let rowOffset = 0;
        for (const block of packed.compressedBlocks) {
            const blockRows = block.dims.at(-2) ?? 0;
            writeBlockIntoBuffer(block, out, outStride, rowOffset, leading, headDim);
            rowOffset += blockRows;
        }

        // Copy dense tail rows (no dequantization needed).
        if (packed.tailRows > 0 && packed.tailData) {
            const tailStride = packed.tailRows * headDim;
            const tailByteOffset = rowOffset * headDim;
            for (let i = 0; i < leading; ++i) {
                out.set(
                    packed.tailData.subarray(i * tailStride, (i + 1) * tailStride),
                    i * outStride + tailByteOffset,
                );
            }
        }

        const combined = new Tensor('float32', out, outDims);
        return packed.originalType !== 'float32' ? combined.to(packed.originalType) : combined;
    }

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

    if (packed.format === 'hybrid') {
        return getPackedEntrySize(packed.compressed) + getPackedEntrySize(packed.tail);
    }

    if (packed.format === 'incremental-hybrid') {
        let total = getTypedArrayByteLength(packed.tailData);
        for (const block of packed.compressedBlocks) {
            total += getPackedEntrySize(block);
        }
        return total;
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
        super();
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
                denseBytes += getTensorByteLength(value);
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
            b_key: 4,
            b_value: 8,
            residual_length: 64,
            rotation_seed: 1337,
            // 'sigma' (mean±k·σ clipping) fits Gaussian post-rotation data and
            // is the Lloyd-Max-style improvement from the TurboQuant paper; use
            // it for long contexts (>= 256 tokens) where the distribution is
            // well-sampled. Fall back to 'minmax' for small-window or short
            // contexts, where σ estimates are noisy.
            quantization: 'minmax', // 'sigma' | 'minmax'
            sigma_k: 2.5,
            residual_correction: true,
            // When the dense tail outgrows residual_length by this many rows,
            // evict that many rows in one packed block. Packing rows in batches
            // of eviction_batch amortizes per-step pack cost and keeps the
            // total number of blocks bounded at N / eviction_batch, so that
            // `materialize()` stays O(N) per call rather than O(N²) over the
            // course of generation.
            eviction_batch: null, // defaults to residual_length when null
            ...config,
        };
        /** @type {Record<string, any>} */
        this.entries = Object.create(null);
        this.seq_length = 0;
    }

    get_seq_length() {
        return this.seq_length;
    }

    /**
     * Pack decoder outputs into CPU-backed buffers.
     *
     * Incremental strategy: when a prior entry exists with seq_length <= the
     * new tensor's seq_length, only the NEW rows are touched — they are
     * appended to a dense tail. When the tail exceeds residual_length, the
     * oldest rows are compressed (rotated + σ-clip quantized + optional QJL
     * residual correction) and stored as an immutable block. All previously
     * compressed blocks are shared by reference with the parent cache.
     * @param {Object} decoderResults Decoder outputs.
     * @param {{ disposeSourceDecoderResults?: boolean }} [options]
     * @returns {TurboQuantCache}
     */
    async update(decoderResults, options = {}) {
        const next = new TurboQuantCache(this.config);

        for (const name in decoderResults) {
            if (!name.startsWith('present')) continue;

            const pastName = getPastKeyValuesName(name);
            const tensor = decoderResults[name];
            const bits = pastName.endsWith('.key') ? this.config.b_key : this.config.b_value;
            const wantResidual =
                this.config.residual_correction && pastName.endsWith('.key');
            const prior = this.entries[pastName] ?? null;

            next.entries[pastName] = await this._incrementalUpdate(
                tensor,
                prior,
                bits,
                wantResidual,
            );

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
     * Core incremental pack step. Returns a packed entry (either
     * incremental-hybrid or a fallback format for pass-through cases).
     */
    async _incrementalUpdate(tensor, prior, bits, wantResidual) {
        const residualLength = this.config.residual_length ?? 0;
        const seed = this.config.rotation_seed ?? 1337;
        const quantization = this.config.quantization;
        const sigmaK = this.config.sigma_k;

        // Pass-through cases: dense storage only.
        // 1. High-bit values don't benefit from quantization.
        // 2. Tensors without a seq dimension can't be incrementally appended.
        if (bits >= 8 || tensor.dims.length < 3) {
            return await packTensorWithResidualWindow(tensor, bits, {
                residualLength,
                seed,
                residualBits: 1,
                residual: wantResidual,
                quantization,
                sigmaK,
            });
        }

        const cpuTensor = tensor.location === 'gpu-buffer' ? await cloneTensorToCPU(tensor) : tensor;
        const originalType = cpuTensor.type;
        const source = cpuTensor.type === 'float32' ? cpuTensor : cpuTensor.to('float32');
        const dims = source.dims.slice();
        const newSeqLength = dims.at(-2) ?? 0;
        const headDim = dims.at(-1) ?? 0;
        const leading = computeLeading(dims);

        // Determine whether we can reuse prior state (incremental) or need a
        // fresh pack. We can reuse if the prior entry is incremental-hybrid,
        // its shape matches, and the new seq is a superset.
        const canIncrement =
            prior &&
            prior.format === 'incremental-hybrid' &&
            prior.headDim === headDim &&
            prior.leading === leading &&
            prior.seqLength <= newSeqLength;

        if (!canIncrement) {
            // Fresh entry: initialize an incremental-hybrid state and pack
            // everything beyond residual_length into a single block.
            return await this._initializeEntry(source, dims, bits, wantResidual, originalType);
        }

        // Incremental path: extract only the delta rows and append.
        const delta = newSeqLength - prior.seqLength;
        if (delta === 0) {
            // Already up-to-date. Return prior (shared by reference — safe
            // because incremental-hybrid state is write-once-per-block).
            return prior;
        }

        const newRowsData = extractSeqSlice(
            source.data,
            leading,
            newSeqLength,
            headDim,
            newSeqLength - delta,
            newSeqLength,
        );

        const mergedTail = appendRowsToTail(
            prior.tailData ?? new Float32Array(0),
            prior.tailRows,
            newRowsData,
            delta,
            leading,
            headDim,
        );
        const mergedRows = prior.tailRows + delta;

        // Evict in batches: once the tail reaches residual_length + eviction_batch,
        // pack eviction_batch rows at once. This bounds the number of compressed
        // blocks to N / eviction_batch, which keeps materialize() O(N) per call
        // instead of O(N²) over the course of generation.
        const evictionBatch = Math.max(1, this.config.eviction_batch ?? residualLength);
        const evictionThreshold = residualLength + evictionBatch;
        let compressedBlocks = prior.compressedBlocks;
        let compressedRows = prior.compressedRows;
        let tailData = mergedTail;
        let tailRows = mergedRows;
        if (mergedRows >= evictionThreshold) {
            // Evict enough rows to bring tail back down to residualLength.
            const evictRows = mergedRows - residualLength;
            const { evicted, remaining, remainingRows } = takeOldestRows(
                mergedTail,
                mergedRows,
                evictRows,
                leading,
                headDim,
            );
            const evictedDims = dims.slice();
            evictedDims[evictedDims.length - 2] = evictRows;
            const evictedTensor = new Tensor(source.type, evicted, evictedDims);
            const block = await packTurboQuantTensor(evictedTensor, bits, {
                seed,
                residualBits: 1,
                residual: wantResidual,
                quantization,
                sigmaK,
            });
            compressedBlocks = [...prior.compressedBlocks, block];
            compressedRows += evictRows;
            tailData = remaining;
            tailRows = remainingRows;
        }

        return {
            format: 'incremental-hybrid',
            dims,
            originalType: prior.originalType,
            seqLength: newSeqLength,
            headDim,
            leading,
            compressedBlocks,
            compressedRows,
            tailData,
            tailRows,
        };
    }

    /**
     * Build a fresh incremental-hybrid entry from a full present.* tensor.
     * If seqLength exceeds residual_length, the front is packed as one block.
     */
    async _initializeEntry(source, dims, bits, wantResidual, originalType) {
        const residualLength = this.config.residual_length ?? 0;
        const seed = this.config.rotation_seed ?? 1337;
        const quantization = this.config.quantization;
        const sigmaK = this.config.sigma_k;
        const seqLength = dims.at(-2) ?? 0;
        const headDim = dims.at(-1) ?? 0;
        const leading = computeLeading(dims);

        /** @type {any[]} */
        const compressedBlocks = [];
        let compressedRows = 0;
        /** @type {Float32Array|null} */
        let tailData = null;
        let tailRows = 0;

        if (residualLength <= 0) {
            // No window: compress everything as a single block.
            if (seqLength > 0) {
                const block = await packTurboQuantTensor(source, bits, {
                    seed,
                    residualBits: 1,
                    residual: wantResidual,
                    quantization,
                    sigmaK,
                });
                compressedBlocks.push(block);
                compressedRows = seqLength;
            }
        } else if (seqLength <= residualLength) {
            // All rows fit in the dense window.
            tailData = extractSeqSlice(source.data, leading, seqLength, headDim, 0, seqLength);
            tailRows = seqLength;
        } else {
            // Split: front goes into a compressed block, back into tail.
            const prefixRows = seqLength - residualLength;
            const prefixData = extractSeqSlice(
                source.data,
                leading,
                seqLength,
                headDim,
                0,
                prefixRows,
            );
            const prefixDims = dims.slice();
            prefixDims[prefixDims.length - 2] = prefixRows;
            const prefixTensor = new Tensor(source.type, prefixData, prefixDims);
            const block = await packTurboQuantTensor(prefixTensor, bits, {
                seed,
                residualBits: 1,
                residual: wantResidual,
                quantization,
                sigmaK,
            });
            compressedBlocks.push(block);
            compressedRows = prefixRows;
            tailData = extractSeqSlice(
                source.data,
                leading,
                seqLength,
                headDim,
                prefixRows,
                seqLength,
            );
            tailRows = residualLength;
        }

        return {
            format: 'incremental-hybrid',
            dims: dims.slice(),
            originalType,
            seqLength,
            headDim,
            leading,
            compressedBlocks,
            compressedRows,
            tailData,
            tailRows,
        };
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
        let compressedBlocks = 0;

        for (const packed of Object.values(this.entries)) {
            packedBytes += getPackedEntrySize(packed);
            const dims = packed.dims ?? [];
            const size = dims.reduce((a, b) => a * b, 1);
            const bytesPerElement = getBytesPerElement(packed.originalType ?? packed.type) || 4;
            denseBytes += size * bytesPerElement;
            entries += 1;
            if (packed.format === 'incremental-hybrid') {
                compressedBlocks += packed.compressedBlocks.length;
            }
        }

        return {
            implementation: 'turboquant',
            entries,
            seq_length: this.seq_length,
            packed_bytes: packedBytes,
            dense_bytes: denseBytes,
            compressed_blocks: compressedBlocks,
        };
    }
}

/**
 * @typedef {_TurboQuantCache} TurboQuantCache
 */

export const TurboQuantCache = /** @type {new (config?: Object) => TurboQuantCache} */ (
    /** @type {unknown} */ (_TurboQuantCache)
);
