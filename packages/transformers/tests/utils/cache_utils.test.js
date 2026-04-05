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

    const cache = await new TurboQuantCache({ b_key: 3, b_value: 3 }).update(decoderResults, {
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

    const cache = await new TurboQuantCache({ b_key: 3, b_value: 3 }).update(decoderResults, {
      disposeSourceDecoderResults: false,
    });

    const stats = cache.getStats();
    expect(stats.implementation).toEqual("turboquant");
    expect(stats.entries).toEqual(2);
    expect(stats.seq_length).toEqual(4);
    expect(stats.packed_bytes).toBeGreaterThan(0);
    expect(stats.dense_bytes).toBeGreaterThan(stats.packed_bytes);
  });
});
