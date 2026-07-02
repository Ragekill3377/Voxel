// VoxelVM Java Benchmark — requires Java 21+ with Panama FFM
// javac --enable-preview --release 21 VoxelBench.java
// java  --enable-preview --enable-native-access=ALL-UNNAMED VoxelBench
package com.voxelvm;

import java.lang.foreign.*;
import java.lang.invoke.*;

public class VoxelBench {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIB;
    private static final MethodHandle jitRun, create, destroy, addSeg, loadProg, run, setS, getSF;

    static {
        var libPath = java.nio.file.Path.of("build/libvoxel_c.so");
        if (!libPath.toFile().exists()) {
            libPath = java.nio.file.Path.of("../build/libvoxel_c.so");
        }
        LIB = SymbolLookup.libraryLookup(libPath, Arena.global());
        jitRun = LINKER.downcallHandle(LIB.find("voxel_jit_run").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_DOUBLE));
        create = LINKER.downcallHandle(LIB.find("voxel_engine_create_f64").get(),
            FunctionDescriptor.of(ValueLayout.ADDRESS));
        destroy = LINKER.downcallHandle(LIB.find("voxel_engine_destroy").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        addSeg = LINKER.downcallHandle(LIB.find("voxel_engine_add_segment").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        loadProg = LINKER.downcallHandle(LIB.find("voxel_engine_load_program").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
        run = LINKER.downcallHandle(LIB.find("voxel_engine_run").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));
        setS = LINKER.downcallHandle(LIB.find("voxel_engine_set_scalar").get(),
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        getSF = LINKER.downcallHandle(LIB.find("voxel_engine_get_scalar_f64").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE, ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
    }

    private static int enc(int op, int rd, int ra, int rb, int imm) {
        return (op & 0xFF) | ((rd & 0xF) << 8) | ((ra & 0xF) << 12) | ((rb & 0xF) << 16) | ((imm & 0xFFF) << 20);
    }

    public static void main(String[] args) throws Throwable {
        int N = 1_000_000;
        int kL = 4; // Engine<f64>::kLanes
        double threshold = 500.0;

        try (var arena = Arena.ofConfined()) {
            // Allocate data off-heap
            var dataSeg = arena.allocate(N * 8);
            var rng = new java.util.Random(42);
            double expectedSum = 0;
            for (int i = 0; i < N; i++) {
                double v = rng.nextDouble() * 1000.0;
                dataSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, v);
                if (v > threshold) expectedSum += v;
            }

            // Build bytecode
            int[] codeRaw = {
                enc(0x70, 0, 1, 0, 0),  // VLOAD
                enc(0xC5, 1, 0, 3, 0),  // VFILTER_GT
                enc(0xD0, 5, 1, 0, 0),  // VSUM
                enc(0x2A, 0, 0, 5, 0),  // ADDF
                enc(0x20, 1, 1, kL, 0), // ADD
                enc(0x40, 1, 2, 0, 0),  // CMP
                enc(0x52, 0, 0, (-6) & 0xFFF, 0), // JNZ
                enc(0x01, 0, 0, 0, 0),  // HALT
            };
            var codeSeg = arena.allocate(codeRaw.length * 4);
            for (int i = 0; i < codeRaw.length; i++) {
                codeSeg.setAtIndex(ValueLayout.JAVA_INT, i, codeRaw[i]);
            }

            System.out.printf("Dataset: %d f64, threshold %.0f, expected %.3f\n", N, threshold, expectedSum);
            System.out.flush();

            // Quick test with 8-element array first
            System.out.println("Quick test..."); System.out.flush();
            try (var smallArena = Arena.ofConfined()) {
                var testData = smallArena.allocate(8 * 8);
                double[] testVals = {1,2,3,4,5,6,7,8};
                for (int i = 0; i < 8; i++) testData.setAtIndex(ValueLayout.JAVA_DOUBLE, i, testVals[i]);
                double testResult = (double) jitRun.invoke(codeSeg, (long)codeRaw.length, testData, 8L, 4.0);
                System.out.printf("  8-element test: %.1f (expected 26.0) %s\n", testResult,
                    Math.abs(testResult - 26.0) < 1e-6 ? "OK" : "FAIL");
                System.out.flush();
            }
            System.out.println("Quick test done"); System.out.flush();

            // Full benchmark
            System.out.println("Warmup..."); System.out.flush();
            jitRun.invoke(codeSeg, (long) codeRaw.length, dataSeg, (long) N, threshold);
            System.out.println("Warmup done"); System.out.flush();

            // Warmup
            jitRun.invoke(codeSeg, (long) codeRaw.length, dataSeg, (long) N, threshold);

            // Measured
            for (int i = 0; i < 3; i++) {
                long t0 = System.nanoTime();
                double result = (double) jitRun.invoke(codeSeg, (long) codeRaw.length, dataSeg, (long) N, threshold);
                long t1 = System.nanoTime();
                double us = (t1 - t0) / 1000.0;
                double mps = N / (us / 1e6) / 1e6;
                boolean ok = Math.abs(result - expectedSum) < expectedSum * 1e-12;
                System.out.printf("  Java JIT run %d: %.0f us, %.0f M/s, %s\n", i+1, us, mps, ok ? "OK" : "FAIL");
            }
        }
    }
}
