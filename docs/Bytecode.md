# Bytecode Reference

## Instruction Format

32-bit packed word: `[opcode:8][rd:4][ra:4][rb:4][imm12:12]`

| Field | Bits | Description |
|-------|------|-------------|
| opcode | 7:0 | 8-bit operation identifier |
| rd | 11:8 | Destination register (0-15 for vectors, 0-63 for scalars) |
| ra | 15:12 | Source register A |
| rb | 19:16 | Source register B |
| imm12 | 31:20 | 12-bit immediate or extended field |

For vector loads and stores, imm12 is subdivided: bits 11:8 encode the segment ID (0-15), bits 7:0 encode the element count (0 = fill the full vector). For VFILTER, bits 2:0 encode the comparison mode (0=EQ, 1=NE, 2=LT, 3=LE, 4=GT, 5=GE). Branch offsets are 12-bit signed instruction offsets (not byte offsets).

## Opcode Table (208 total)

### Control (0x00-0x0F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| NOP | 0x00 | No operation |
| HALT | 0x01 | Stop execution |
| TRAP | 0x02 | Trap with error flag |
| BREAK | 0x03 | Debug breakpoint |
| YIELD | 0x04 | Yield to scheduler |
| BARRIER | 0x05 | Memory barrier |
| PREFETCH | 0x06 | Prefetch segment data |
| FLUSH_CACHE | 0x07 | Flush CPU cache |
| SYNC | 0x08 | Thread sync barrier |
| MEMFENCE | 0x09 | Full memory fence |

### Scalar Move / Immediate (0x10-0x1F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| MOV | 0x10 | Rd = sign-extend(imm12) |
| MOVR | 0x11 | Rd = Ra (copy) |
| ADDI | 0x12 | Rd = Ra + sign-extend(imm12) |
| SUBI | 0x13 | Rd = Ra - sign-extend(imm12) |
| MULI | 0x14 | Rd = Ra * sign-extend(imm12) |
| ANDI | 0x15 | Rd = Ra & imm12 |
| ORI | 0x16 | Rd = Ra \| imm12 |
| XORI | 0x17 | Rd = Ra ^ imm12 |
| SHLI | 0x18 | Rd = Ra << (imm12 & 0x3F) |
| SHRI | 0x19 | Rd = Ra >> (imm12 & 0x3F) |
| SAR_I | 0x1A | Rd = Ra >>a (imm12 & 0x3F) |
| MOVZ | 0x1B | Conditional move (zero) |
| MOVN | 0x1C | Conditional move (not zero) |
| MOVK | 0x1D | Move 16-bit field |
| LEA | 0x1E | Rd = Ra + sign-extend(imm12) |

### Scalar Arithmetic (0x20-0x2F)

| Opcode | Hex | Domain |
|--------|-----|--------|
| ADD | 0x20 | i64 |
| SUB | 0x21 | i64 |
| MUL | 0x22 | i64 |
| DIV | 0x23 | i64 |
| MOD | 0x24 | i64 |
| NEG | 0x25 | i64 |
| ABS | 0x26 | i64 |
| MIN | 0x27 | i64 |
| MAX | 0x28 | i64 |
| AVG | 0x29 | i64 |
| ADDF | 0x2A | f64 |
| SUBF | 0x2B | f64 |
| MULF | 0x2C | f64 |
| DIVF | 0x2D | f64 |
| NEGF | 0x2E | f64 |
| ABSF | 0x2F | f64 |

### Scalar Bitwise (0x30-0x3F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| AND | 0x30 | Rd = Ra & Rb |
| OR | 0x31 | Rd = Ra \| Rb |
| XOR | 0x32 | Rd = Ra ^ Rb |
| NOT | 0x33 | Rd = ~Ra |
| SHL | 0x34 | Rd = Ra << Rb |
| SHR | 0x35 | Rd = Ra >> Rb |
| SAR | 0x36 | Rd = Ra >>a Rb |
| ROL | 0x37 | Rd = Ra rotl Rb |
| ROR | 0x38 | Rd = Ra rotr Rb |
| POPCNT | 0x39 | Rd = popcount(Ra) |
| CLZ | 0x3A | Rd = count-leading-zeros(Ra) |
| CTZ | 0x3B | Rd = count-trailing-zeros(Ra) |
| BSWAP | 0x3C | Rd = byte-swap(Ra) |
| BEXTR | 0x3D | Rd = bit-extract(Ra, Rb) |
| BZHI | 0x3E | Rd = zero-high-bits(Ra, Rb) |
| PDEP | 0x3F | Rd = parallel-deposit(Ra, Rb) |

### Comparison (0x40-0x4F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| CMP | 0x40 | Compare Ra, Rb (i64), sets flags |
| CMPF | 0x41 | Compare Ra, Rb (f64), sets flags |
| CMPU | 0x42 | Compare Ra, Rb (u64), sets flags |
| TST | 0x43 | AND Ra, Rb, set flags on result |
| TSTF | 0x44 | Compare Ra, Rb (f64), set numeric flags |
| ISNULL | 0x45 | Rd = (Ra == 0) ? 1 : 0 |
| ISNOTNULL | 0x46 | Rd = (Ra != 0) ? 1 : 0 |
| SELECT | 0x47 | Rd = Zero ? Ra : Rb |
| SELECTV | 0x48 | Vd = mask[lane] ? Vb : Va |

### Branching (0x50-0x5F)

| Opcode | Hex | Condition |
|--------|-----|-----------|
| JMP | 0x50 | Unconditional |
| JZ | 0x51 | Zero flag set |
| JNZ | 0x52 | Zero flag clear |
| JS | 0x53 | Sign flag set |
| JNS | 0x54 | Sign flag clear |
| JO | 0x55 | Overflow flag set |
| JNO | 0x56 | Overflow flag clear |
| JC | 0x57 | Carry flag set |
| JNC | 0x58 | Carry flag clear |
| JL | 0x59 | Less than |
| JLE | 0x5A | Less or equal |
| JG | 0x5B | Greater than |
| JGE | 0x5C | Greater or equal |
| CALL | 0x5D | Subroutine call |
| RET | 0x5E | Return from call |
| TABLE_JMP | 0x5F | PC = base[Ra] + Rb |

### Type Conversion (0x60-0x6F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| CVT_I8 | 0x60 | Rd = (i8)Ra |
| CVT_I16 | 0x61 | Rd = (i16)Ra |
| CVT_I32 | 0x62 | Rd = (i32)Ra |
| CVT_I64 | 0x63 | Rd = (i64)Ra |
| CVT_F32 | 0x64 | Rd = (f32)Ra |
| CVT_F64 | 0x65 | Rd = (f64)Ra |
| CVT_U8 | 0x66 | Rd = (u8)Ra |
| CVT_U16 | 0x67 | Rd = (u16)Ra |
| CVT_U32 | 0x68 | Rd = (u32)Ra |
| CVT_U64 | 0x69 | Rd = (u64)Ra |
| BITCAST | 0x6A | Rd = Ra (raw u64 copy) |
| REINTERPRET | 0x6B | Rd = T(Ra) via reinterpret |
| TRUNC | 0x6C | Rd = trunc(Ra) |
| ROUND | 0x6D | Rd = round(Ra) |
| CEIL | 0x6E | Rd = ceil(Ra) |
| FLOOR | 0x6F | Rd = floor(Ra) |

### Vector I/O (0x70-0x7F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VLOAD | 0x70 | Vd[i] = seg[Ra + i] |
| VSTORE | 0x71 | seg[Ra + i] = Vs[i] |
| VGATHER | 0x72 | Vd[i] = seg[indices[i]] |
| VSCATTER | 0x73 | seg[indices[i]] = Vs[i] |
| VLOAD_STRIDED | 0x74 | Vd[i] = seg[Ra + i*Rb] |
| VSTORE_STRIDED | 0x75 | seg[Ra + i*Rb] = Vs[i] |
| VLOAD_MASKED | 0x76 | Vd[i] = mask[i] ? seg[Ra+i] : Vd[i] |
| VSTORE_MASKED | 0x77 | mask[i] ? seg[Ra+i] = Vs[i] |
| VSPLAT | 0x78 | Vd[i] = Ra (broadcast) |
| VEXTRACT | 0x79 | Rd = Va[lane] |
| VINSERT | 0x7A | Vd[lane] = Rb |
| VPERMUTE | 0x7B | Vd[i] = Va[idxVec[i]] |
| VSHUFFLE | 0x7C | Vd = shuffle(Va, Vb, mode) |
| VREVERSE | 0x7D | Vd[i] = Va[N-1-i] |
| VROTATE | 0x7E | Vd[i] = Va[(i+sh) % N] |
| VSLIDE | 0x7F | Vd[i] = Va[i+sh] |

### Vector Arithmetic (0x80-0x8F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VADD | 0x80 | Vd = Va + Vb |
| VSUB | 0x81 | Vd = Va - Vb |
| VMUL | 0x82 | Vd = Va * Vb |
| VDIV | 0x83 | Vd = Va / Vb |
| VMOD | 0x84 | Vd = Va % Vb |
| VNEG | 0x85 | Vd = -Va |
| VABS | 0x86 | Vd = \|Va\| |
| VMIN | 0x87 | Vd = min(Va, Vb) |
| VMAX | 0x88 | Vd = max(Va, Vb) |
| VAVG | 0x89 | Vd = (Va+Vb)/2 |
| VFMA | 0x8A | Vd = Va*Vb + c |
| VFMS | 0x8B | Vd = Va*Vb - c |
| VSQRT | 0x8C | Vd = sqrt(Va) |
| VRSQRT | 0x8D | Vd = 1/sqrt(Va) |
| VRCP | 0x8E | Vd = 1/Va |
| VPOW | 0x8F | Vd = Va ^ Vb |

### Vector-Scalar (0x90-0x9F)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VSADD | 0x90 | Vd = Va + Rb |
| VSSUB | 0x91 | Vd = Va - Rb |
| VSMUL | 0x92 | Vd = Va * Rb |
| VSDIV | 0x93 | Vd = Va / Rb |
| VSMOD | 0x94 | Vd = Va % Rb |
| VSMIN | 0x95 | Vd = min(Va, Rb) |
| VSMAX | 0x96 | Vd = max(Va, Rb) |

### Vector Compare (0xA0-0xAF)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VCMPEQ | 0xA0 | Vd = Va == Vb |
| VCMPNE | 0xA1 | Vd = Va != Vb |
| VCMPLT | 0xA2 | Vd = Va < Vb |
| VCMPLE | 0xA3 | Vd = Va <= Vb |
| VCMPGT | 0xA4 | Vd = Va > Vb |
| VCMPGE | 0xA5 | Vd = Va >= Vb |
| VCMPNULL | 0xA6 | Vd = Va is null |
| VCMPNOTNULL | 0xA7 | Vd = Va is not null |

### Vector Logical (0xB0-0xBF)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VAND | 0xB0 | Vd = Va & Vb |
| VOR | 0xB1 | Vd = Va \| Vb |
| VXOR | 0xB2 | Vd = Va ^ Vb |
| VNOT | 0xB3 | Vd = ~Va |
| VANDN | 0xB4 | Vd = ~Va & Vb |
| VSHL | 0xB5 | Vd = Va << Vb |
| VSHR | 0xB6 | Vd = Va >> Vb |
| VSAR | 0xB7 | Vd = Va >>a Vb |

### Vector Filter (0xC0-0xCF)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VFILTER | 0xC0 | Vd = Va[pred](Rb) |
| VFILTER_EQ | 0xC1 | Vd = Va if Va == Rb |
| VFILTER_NE | 0xC2 | Vd = Va if Va != Rb |
| VFILTER_LT | 0xC3 | Vd = Va if Va < Rb |
| VFILTER_LE | 0xC4 | Vd = Va if Va <= Rb |
| VFILTER_GT | 0xC5 | Vd = Va if Va > Rb |
| VFILTER_GE | 0xC6 | Vd = Va if Va >= Rb |
| VBLEND | 0xC7 | Vd = mask ? Vb : Va |
| VMASK_STORE | 0xC8 | seg[Ra+i] = Vs[i] if mask[i] |
| VMASK_LOAD | 0xC9 | Vd[i] = seg[Ra+i] if mask[i] |

### Vector Reduction (0xD0-0xDF)

| Opcode | Hex | Description |
|--------|-----|-------------|
| VSUM | 0xD0 | Rd = sum(Va) |
| VPROD | 0xD1 | Rd = product(Va) |
| VMEAN | 0xD2 | Rd = mean(Va) |
| VSTDDEV | 0xD3 | Rd = stddev(Va) |
| VVARIANCE | 0xD4 | Rd = variance(Va) |
| VRED_MIN | 0xD5 | Rd = min(Va) |
| VRED_MAX | 0xD6 | Rd = max(Va) |
| VCOUNT | 0xD7 | Rd = count non-zero(Va) |
| VANY | 0xD8 | Rd = any non-zero(Va) |
| VALL | 0xD9 | Rd = all non-zero(Va) |
| VFIRST | 0xDA | Rd = Va[0] |
| VLAST | 0xDB | Rd = Va[N-1] |
| VNTH | 0xDC | Rd = Va[n] |
| WDELTA | 0xDD | Vd[i] = seg[i+1] - seg[i], streaming over segment |
| WINDOW_SUM | 0xDE | Vd[i] = sum(seg[i..i+win-1]), sliding window sum |
| WINDOW_MEAN | 0xDF | Vd[i] = WINDOW_SUM / window, sliding window mean |

Window-streaming ops (0xDD-0xDF) read from an engine segment and fill a vector register with kLanes windowed results. They advance an offset scalar register by kLanes on each call, designed to run in a tight bytecode loop with VSTORE to materialize the full output array. See `Engine::WindowDelta/WindowSum/WindowMean` for the direct static-method API that computes the full array in one call.

### Aggregate (0xE0-0xEF)

| Opcode | Hex | Description |
|--------|-----|-------------|
| AGG_COUNT | 0xE0 | Count non-zero in seg |
| AGG_SUM | 0xE1 | Sum seg |
| AGG_AVG | 0xE2 | Average seg |
| AGG_MIN | 0xE3 | Min seg |
| AGG_MAX | 0xE4 | Max seg |
| AGG_FIRST | 0xE5 | First value |
| AGG_LAST | 0xE6 | Last value |
| AGG_STDDEV | 0xE7 | StdDev (Welford) |
| AGG_VARIANCE | 0xE8 | Variance (Welford) |
| AGG_COUNT_DISTINCT | 0xE9 | Count distinct |
| AGG_SUM_DISTINCT | 0xEA | Sum distinct |
| AGG_MEDIAN | 0xEB | Median |
| AGG_MODE | 0xEC | Mode |
| AGG_PERCENTILE | 0xED | Percentile |
| HASH_INIT | 0xEE | Init hash table |
| HASH_PROBE | 0xEF | Probe hash table |

### Hash / Sort / Join (0xF0-0xFF)

| Opcode | Hex | Description |
|--------|-----|-------------|
| HASH_BUILD | 0xF0 | Build hash table |
| HASH_LOOKUP | 0xF1 | Lookup key |
| SORT_ASC | 0xF2 | Sort ascending |
| SORT_DESC | 0xF3 | Sort descending |
| SORT_TOPK | 0xF4 | Partial sort top-K |
| SORT_BOTTOMK | 0xF5 | Partial sort bottom-K |
| JOIN_HASH | 0xF6 | Hash join |
| JOIN_MERGE | 0xF7 | Merge join |
| JOIN_NESTED | 0xF8 | Nested loop join |
| JOIN_ANTI | 0xF9 | Anti join |
| JOIN_SEMI | 0xFA | Semi join |
| WINDOW_ROW | 0xFB | Row window |
| WINDOW_RANGE | 0xFC | Range window |
| PARTITION_HASH | 0xFD | Hash partition |
| SERIALIZE | 0xFE | Serialize segment |
| DESERIALIZE | 0xFF | Deserialize segment |
