// VoxelVM Java Bindings — Project Panama FFM API (Java 21+)
// Zero-copy, no JNI. Requires: --enable-preview --enable-native-access=ALL-UNNAMED
package com.voxelvm;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;

public class VoxelVM {
    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIB;
    
    static {
        // Load libvoxel_c.so from the build directory or system path
        Path libPath = Path.of(System.getProperty("user.dir"), "build", "libvoxel_c.so");
        if (!libPath.toFile().exists()) {
            libPath = Path.of("build", "libvoxel_c.so");
        }
        LIB = SymbolLookup.libraryLookup(libPath, Arena.global());
    }

    // ================================================================
    // Engine
    // ================================================================
    public static class Engine implements AutoCloseable {
        private final MemorySegment ptr;
        private static final MethodHandle create, destroy, addSeg, loadProg, run, setS, getS, setSF, getSF, runPar;
        static {
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
        }

        public Engine() throws Throwable {
            ptr = (MemorySegment) create.invokeExact();
        }

        public int addSegment(MemorySegment data, long count) throws Throwable {
            return (int)(long) addSeg.invokeExact(ptr, data, count);
        }

        public void loadProgram(MemorySegment code, long len) throws Throwable {
            loadProg.invokeExact(ptr, code, len);
        }

        public void run() throws Throwable { run.invokeExact(ptr); }

        @Override public void close() throws Throwable { destroy.invokeExact(ptr); }
    }

    // ================================================================
    // JIT — one-shot filter+sum
    // ================================================================
    private static final MethodHandle jitRun;
    static {
        jitRun = LINKER.downcallHandle(LIB.find("voxel_jit_run").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_DOUBLE));
    }

    public static double jitRun(MemorySegment code, long codeLen,
                                MemorySegment data, long dataLen, double threshold) throws Throwable {
        return (double) jitRun.invokeExact(code, codeLen, data, dataLen, threshold);
    }

    // ================================================================
    // Usage example (main method)
    // ================================================================
    public static void main(String[] args) throws Throwable {
        try (var arena = Arena.ofConfined()) {
            double[] input = {1, 2, 3, 4, 5, 6, 7, 8};
            long N = input.length;

            // Allocate off-heap, zero copy
            var dataSeg = arena.allocate(N * 8);
            for (int i = 0; i < N; i++) {
                dataSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, input[i]);
            }

            // Build bytecode
            int[] codeRaw = instrBuf();
            var codeSeg = arena.allocate(codeRaw.length * 4);
            for (int i = 0; i < codeRaw.length; i++) {
                codeSeg.setAtIndex(ValueLayout.JAVA_INT, i, codeRaw[i]);
            }

            // JIT
            double result = jitRun(codeSeg, codeRaw.length, dataSeg, N, 4.0);
            System.out.printf("JIT result: %.1f (expected 26.0)\n", result);
        }
    }

    private static int[] instrBuf() {
        int kVLoad = 0x70, kVFilterGt = 0xC5, kVSum = 0xD0, kAddf = 0x2A, kAdd = 0x20, kCmp = 0x40, kJnz = 0x52, kHalt = 0x01;
        int kLanes = 4;
        return new int[]{
            enc(kVLoad, 0, 1, 0, 0),
            enc(kVFilterGt, 1, 0, 3),
            enc(kVSum, 5, 1),
            enc(kAddf, 0, 0, 5),
            enc(kAdd, 1, 1, kLanes),
            enc(kCmp, 1, 2),
            enc(kJnz, 0, 0, (-6) & 0xFFF),
            enc(kHalt, 0, 0, 0),
        };
    }

    private static int enc(int op, int rd, int ra, int rb, int imm) {
        return (op & 0xFF) | ((rd & 0xF) << 8) | ((ra & 0xF) << 12) | ((rb & 0xF) << 16) | ((imm & 0xFFF) << 20);
    }
}
