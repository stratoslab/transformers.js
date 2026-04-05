import { init } from "../init.js";
import { DynamicCache, TurboQuantCache } from "../../src/cache_utils.js";
import { Tensor } from "../../src/utils/tensor.js";

init();

function makeKV(type = "float32") {
  const dims = [1, 2, 4, 8];
  const values = Float32Array.from(
    { length: dims.reduce((a, b) => a * b, 1) },
    (_, i) => Math.sin(i / 3) * 0.75 + Math.cos(i / 5) * 0.25,
  );
  return new Tensor("float32", values, dims).to(type);
}

describe("Cache utilities", () => {
  it("reports dense cache stats for DynamicCache", () => {
    const key = makeKV();
    const value = makeKV();
    const cache = new DynamicCache({
      "past_key_values.0.key": key,
      "past_key_values.0.value": value,
    });

    const stats = cache.getStats();
    expect(stats.implementation).toEqual("dynamic");
    expect(stats.entries).toEqual(2);
    expect(stats.seq_length).toEqual(4);
    expect(stats.packed_bytes).toEqual(stats.dense_bytes);
    expect(stats.packed_bytes).toBeGreaterThan(0);
  });

  it("packs TurboQuant cache entries and preserves tensor shapes", async () => {
    const decoderResults = {
      "present.0.key": makeKV(),
      "present.0.value": makeKV(),
    };

    const cache = await new TurboQuantCache({ b_key: 3, b_value: 3, residual_length: 0 }).update(decoderResults, {
      disposeSourceDecoderResults: false,
    });

    expect(cache.get_seq_length()).toEqual(4);

    const materialized = cache.materialize();
    expect(materialized["past_key_values.0.key"].dims).toEqual([1, 2, 4, 8]);
    expect(materialized["past_key_values.0.value"].dims).toEqual([1, 2, 4, 8]);

    expect(materialized["past_key_values.0.key"].tolist()).toBeCloseToNested(
      decoderResults["present.0.key"].tolist(),
      0,
    );
    expect(materialized["past_key_values.0.value"].tolist()).toBeCloseToNested(
      decoderResults["present.0.value"].tolist(),
      0,
    );
  });

  it("reports compressed size smaller than dense size for TurboQuantCache", async () => {
    const decoderResults = {
      "present.0.key": makeKV("float16"),
      "present.0.value": makeKV("float16"),
    };

    const cache = await new TurboQuantCache({ b_key: 3, b_value: 3, residual_length: 0 }).update(decoderResults, {
      disposeSourceDecoderResults: false,
    });

    const stats = cache.getStats();
    expect(stats.implementation).toEqual("turboquant");
    expect(stats.entries).toEqual(2);
    expect(stats.seq_length).toEqual(4);
    expect(stats.packed_bytes).toBeGreaterThan(0);
    expect(stats.dense_bytes).toBeGreaterThan(stats.packed_bytes);
  });

  // Build a [1, 2, seqLen, 8] tensor whose value at (lead, row, h) depends
  // only on (lead, row, h) — so tensors of different lengths agree at their
  // shared positions. This is what the decoder actually emits: each step
  // produces a cumulative present.* tensor where past positions are stable.
  function makeGrowingKV(seqLen) {
    const data = new Float32Array(2 * seqLen * 8);
    let i = 0;
    for (let lead = 0; lead < 2; ++lead) {
      for (let row = 0; row < seqLen; ++row) {
        for (let h = 0; h < 8; ++h) {
          const k = lead * 1000 + row * 37 + h;
          data[i++] = Math.sin(k / 3) * 0.75 + Math.cos(k / 5) * 0.25;
        }
      }
    }
    return new Tensor("float32", data, [1, 2, seqLen, 8]);
  }

  it("appends incrementally without repacking the prefix", async () => {
    // Simulate 1→12 token generation. The cache must end up with the
    // full seq_length and values close to the original tensor.
    let cache = new TurboQuantCache({ b_key: 3, b_value: 3, residual_length: 4 });
    for (let n = 1; n <= 12; ++n) {
      cache = await cache.update({
        "present.0.key": makeGrowingKV(n),
        "present.0.value": makeGrowingKV(n),
      });
    }
    expect(cache.get_seq_length()).toEqual(12);

    const materialized = cache.materialize();
    expect(materialized["past_key_values.0.key"].dims).toEqual([1, 2, 12, 8]);
    expect(materialized["past_key_values.0.value"].dims).toEqual([1, 2, 12, 8]);

    expect(materialized["past_key_values.0.key"].tolist()).toBeCloseToNested(
      makeGrowingKV(12).tolist(),
      0,
    );

    // Recent tokens (inside the dense tail) must round-trip exactly.
    const truth = makeGrowingKV(12).data;
    const got = materialized["past_key_values.0.key"].data;
    // Last 4 rows sit in the tail. For leading index i, they live at
    // [i * 12 * 8 + 8 * 8 .. i * 12 * 8 + 12 * 8).
    for (let lead = 0; lead < 2; ++lead) {
      for (let row = 8; row < 12; ++row) {
        for (let h = 0; h < 8; ++h) {
          const flat = lead * 96 + row * 8 + h;
          expect(got[flat]).toBeCloseTo(truth[flat], 6);
        }
      }
    }
  });

  it("honors eviction_batch for controlling block count", async () => {
    // residual_length=4, eviction_batch=8 → eviction fires whenever the tail
    // reaches residual+batch=12, packing 8 rows at a time.
    // Generating 20 tokens: evicts at seq=12 (block 1, 8 rows) and seq=20
    // (block 2, 8 rows). Final state: 2 blocks + 4-row tail.
    let cache = new TurboQuantCache({
      b_key: 3,
      b_value: 3,
      residual_length: 4,
      eviction_batch: 8,
    });
    for (let n = 1; n <= 20; ++n) {
      cache = await cache.update({ "present.0.key": makeGrowingKV(n) });
    }
    const stats = cache.getStats();
    expect(stats.seq_length).toEqual(20);
    expect(stats.compressed_blocks).toEqual(2);

    // Same generation length with batch=1 (the worst-case, finest-grained
    // eviction) should create O(seq) blocks instead.
    let fineGrained = new TurboQuantCache({
      b_key: 3,
      b_value: 3,
      residual_length: 4,
      eviction_batch: 1,
    });
    for (let n = 1; n <= 20; ++n) {
      fineGrained = await fineGrained.update({ "present.0.key": makeGrowingKV(n) });
    }
    // First 5 steps fill the tail without eviction, next 15 each trigger one.
    expect(fineGrained.getStats().compressed_blocks).toEqual(16);
  });
});
