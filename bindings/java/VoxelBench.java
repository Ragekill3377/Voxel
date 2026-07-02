// VoxelVM Java Benchmark — requires Java 21+ with Panama FFM
// javac --enable-preview --release 21 VoxelBench.java
// java  --enable-preview --enable-native-access=ALL-UNNAMED VoxelBench
import java.lang.foreign.*;
import java.lang.invoke.*;
import java.nio.file.Path;
import java.util.*;

public class VoxelBench {
    static int E(int o,int r,int a,int b,int i){return (o&0xFF)|((r&0xF)<<8)|((a&0xF)<<12)|((b&0xF)<<16)|((i&0xFFF)<<20);}

    public static void main(String[] args) throws Throwable {
        var linker = Linker.nativeLinker();
        String libName = System.getProperty("voxel.lib");
        Path libPath;
        if (libName != null) {
            libPath = Path.of(libName);
        } else {
            libName = System.mapLibraryName("voxel_c");
            libPath = Path.of("build", libName);
            if (!libPath.toFile().exists())
                libPath = Path.of(System.getProperty("user.dir"), "build", libName);
        }
        System.out.println("Loading: " + libPath.toAbsolutePath());
        var lib = SymbolLookup.libraryLookup(libPath, Arena.global());
        var jitRun = linker.downcallHandle(lib.find("voxel_jit_run").get(),
            FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG,
                ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_DOUBLE));

        int N = 1_000_000; double th = 500.0;
        try (var arena = Arena.ofConfined()) {
            var dataSeg = arena.allocate(N * 8);
            var rng = new Random(42);
            double expected = 0;
            for (int i = 0; i < N; i++) {
                double v = rng.nextDouble() * 1000.0;
                dataSeg.setAtIndex(ValueLayout.JAVA_DOUBLE, i, v);
                if (v > th) expected += v;
            }
            int[] raw = {E(0x70,0,1,0,0),E(0xC5,1,0,3,0),E(0xD0,5,1,0,0),E(0x2A,0,0,5,0),
                E(0x20,1,1,4,0),E(0x40,0,1,2,0),E(0x52,0,0,0,(-6)&0xFFF),E(0x01,0,0,0,0)};
            var codeSeg = arena.allocate(raw.length * 4);
            for (int i = 0; i < raw.length; i++) codeSeg.setAtIndex(ValueLayout.JAVA_INT, i, raw[i]);

            System.out.printf("Dataset: %d f64, thresh %.0f, expected %.3f\n", N, th, expected);
            // Warmup
            double d1 = (double) jitRun.invokeExact((MemorySegment)codeSeg, (long)raw.length, (MemorySegment)dataSeg, (long)N, th);
            double d2 = (double) jitRun.invokeExact((MemorySegment)codeSeg, (long)raw.length, (MemorySegment)dataSeg, (long)N, th);

            var times = new ArrayList<Double>();
            for (int i = 0; i < 3; i++) {
                long t0 = System.nanoTime();
                double res = (double) jitRun.invokeExact((MemorySegment)codeSeg, (long)raw.length, (MemorySegment)dataSeg, (long)N, th);
                long t1 = System.nanoTime();
                double us = (t1-t0)/1000.0;
                times.add(us);
                double mps = N / (us/1e6) / 1e6;
                System.out.printf("  Java JIT run %d: %.0f us, %.0f M/s, %s\n", i+1, us, mps, Math.abs(res-expected) < expected * 1e-12?"OK":"FAIL");
            }
            Collections.sort(times);
            double med = times.get(1);
            System.out.printf("  Java JIT median: %.0f us, %.0f M/s\n", med, N/(med/1e6)/1e6);
        }
    }
}
