#include "bindings/voxel_c.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    // Test: filter+sum via JIT
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    size_t N = sizeof(data)/sizeof(data[0]);
    size_t kL = voxel_engine_k_lanes();

    uint32_t code[16];
    size_t ci = 0;
    int16_t neg6 = -6;
    code[ci++] = voxel_instr_vload(0, 1, 0, 0);
    code[ci++] = voxel_instr_vfilter_gt(1, 0, 3);
    code[ci++] = voxel_instr_vsum(5, 1);
    code[ci++] = voxel_instr_addf(0, 0, 5);
    code[ci++] = voxel_instr_add(1, 1, (int16_t)kL);
    code[ci++] = voxel_instr_cmp(1, 2);
    code[ci++] = voxel_instr_jnz(neg6);
    code[ci++] = voxel_instr_halt();

    // Test 1: Interpreter
    void* eng = voxel_engine_create_f64();
    voxel_engine_add_segment(eng, data, N);
    voxel_engine_set_scalar(eng, 0, 0);
    voxel_engine_set_scalar(eng, 1, 0);
    voxel_engine_set_scalar(eng, 2, N);
    voxel_engine_set_scalar_f64(eng, 3, 4.0);
    voxel_engine_load_program(eng, code, ci);
    voxel_engine_run(eng);
    double sum = voxel_engine_get_scalar_f64(eng, 0);
    printf("Interpreter: %.1f (expected 26.0) %s\n", sum, (sum == 26.0) ? "OK" : "FAIL");
    voxel_engine_destroy(eng);

    // Test 2: JIT
    double jitResult = voxel_jit_run(code, ci, data, N, 4.0);
    printf("JIT:         %.1f (expected 26.0) %s\n", jitResult, (jitResult == 26.0) ? "OK" : "FAIL");

    // Test 3: NullBitmap
    void* nb = voxel_null_create(64);
    voxel_null_set_null(nb, 10);
    voxel_null_set_null(nb, 20);
    voxel_null_set_null(nb, 30);
    printf("NullBitmap:  nulls=%zu valid=%zu %s\n",
           voxel_null_null_count(nb), voxel_null_valid_count(nb),
           (voxel_null_null_count(nb) == 3) ? "OK" : "FAIL");
    voxel_null_destroy(nb);

    // Test 4: HashAggregator
    uint32_t keys[] = {0, 1, 0, 1, 0};
    double vals[] = {10, 20, 30, 40, 50};
    void* agg = voxel_agg_create();
    voxel_agg_init(agg, 10);
    voxel_agg_accumulate(agg, keys, vals, 5);
    printf("HashAgg:     groups=%zu %s\n",
           voxel_agg_group_count(agg),
           (voxel_agg_group_count(agg) == 2) ? "OK" : "FAIL");
    voxel_agg_destroy(agg);

    // Test 5: Sort
    double sdata[] = {3, 1, 4, 1, 5};
    uint32_t idx[5];
    voxel_sort_ascending(sdata, 5, idx);
    printf("Sort:        idx[0..4] = [%u %u %u %u %u] %s\n",
           idx[0], idx[1], idx[2], idx[3], idx[4],
           (sdata[idx[0]]==1 && sdata[idx[4]]==5) ? "OK" : "FAIL");

    // Test 6: TopK
    double tkdata[] = {3, 1, 4, 1, 5, 9, 2, 6};
    double top3[3];
    voxel_topk(tkdata, 8, 3, top3, 1);
    printf("TopK:        [%.0f %.0f %.0f] (expected [9 6 5]) %s\n",
           top3[0], top3[1], top3[2],
           (top3[0]==9 && top3[1]==6 && top3[2]==5) ? "OK" : "FAIL");

    printf("\nAll C ABI tests complete!\n");
    return 0;
}
