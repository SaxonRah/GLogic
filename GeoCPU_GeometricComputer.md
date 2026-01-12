# GeoCPU: A Geometric Computer

**What if computation could see correlations that Boolean circuits can't? Boolean circuits can only ask: "Is this true?"** Geometric computers can ask: "Is this true, **and how are these variables correlated**?" This FPGA design implements the first processor operating natively on Clifford algebra multivectors.

Since Boolean logic is a proper subset of Geometric Logic (as proven in our [computational proof](https://github.com/SaxonRah/GLogic/blob/main/Boolean_GLogic.py)), we can build computers that operate on the *full* geometric structure; not just its scalar projection. This means native operations on variable correlations, parallel evaluation of logical relationships, and continuous interpolation between discrete logical states.

Below is an FPGA implementation targeting the **Alchitry Au V2** (XC7A35T Artix-7) using **Lucid V2**. The design implements a complete processor operating on multivectors in Cl(2,0), with a pipelined geometric product unit, four multivector registers, and a custom instruction set.

**Status**: This is a proof-of-concept design that has not been synthesized or tested on hardware. Resource estimates suggest ~7% utilization of the FPGA, with an expected performance of ~33M geometric products/second at 100MHz.

## Why FPGA?

The geometric product requires 16 simultaneous multiplications for n=2 (4D Clifford algebra). FPGAs provide:
- **Parallel DSP slices**: XC7A35T has 50 DSP48E1 units—perfect for concurrent multiplication
- **Custom datapaths**: Multivector operations don't fit CPU architectures  
- **Fixed-point arithmetic**: Q8.8 format gives sufficient precision with hardware efficiency
- **Deterministic timing**: Critical for real-time correlation detection

---

## Resource Analysis

**XC7A35T Resources:**
- 20,800 logic cells
- 50 DSP48E1 slices (18x25-bit multipliers)
- 1.8 Mb block RAM (enough for ~57k 32-bit words)
- 100MHz oscillator

**Multivector sizes:**
- n=2: 4 components (scalar, e₁, e₂, e₁₂) ✓ Very feasible
- n=3: 8 components ✓ Feasible with optimization
- n=4: 16 components ⚠ Tight, but possible

We will target **n=2 initially** (4D Clifford algebra) as proof of concept, with architecture designed to scale to n=3.

## GeoCPU Architecture

```
┌─────────────────────────────────────────────────┐
│              GeoCPU Core (n=2)                  │
├─────────────────────────────────────────────────┤
│  Registers (4x multivectors = 16 components)    │
│    R0: [scalar, e1, e2, e12] (4x 16-bit fixed)  │
│    R1: [scalar, e1, e2, e12]                    │
│    R2: [scalar, e1, e2, e12]                    │
│    R3: [scalar, e1, e2, e12]                    │
├─────────────────────────────────────────────────┤
│  Execution Units                                │
│    • Geometric Product Unit (pipelined)         │
│    • Grade Projection Unit                      │
│    • Scalar Operations Unit                     │
│    • Boolean Embedding Unit                     │
├─────────────────────────────────────────────────┤
│  Memory Interface                               │
│    • Multivector memory (128 entries)           │
│    • Multiplication table ROM                   │
│    • Program ROM (instruction memory)           │
├─────────────────────────────────────────────────┤
│  I/O                                            │
│    • UART interface (for host communication)    │
│    • LED outputs (show register values)         │
│    • Button input (reset/step)                  │
└─────────────────────────────────────────────────┘
```

## Lucid V2 Implementation

### 1. Multivector Data Type

```lucid
// multivector.luc - Basic multivector type for Cl(2,0)

module multivector {
  // Fixed-point Q8.8 format (8 integer bits, 8 fractional bits)
  // Range: -128.00 to +127.996
  
  struct mv2d {
    scalar : $signed[16],  // Component 0: scalar
    e1     : $signed[16],  // Component 1: e₁
    e2     : $signed[16],  // Component 2: e₂  
    e12    : $signed[16]   // Component 3: e₁₂
  }
  
  // Zero multivector
  const ZERO = mv2d{
    scalar: 16h0000,
    e1:     16h0000,
    e2:     16h0000,
    e12:    16h0000
  }
  
  // Basis vectors
  const E1 = mv2d{
    scalar: 16h0000,
    e1:     16h0100,  // 1.0 in Q8.8
    e2:     16h0000,
    e12:    16h0000
  }
  
  const E2 = mv2d{
    scalar: 16h0000,
    e1:     16h0000,
    e2:     16h0100,  // 1.0 in Q8.8
    e12:    16h0000
  }
  
  const ONE = mv2d{
    scalar: 16h0100,  // 1.0 in Q8.8
    e1:     16h0000,
    e2:     16h0000,
    e12:    16h0000
  }
}
```

### 2. Geometric Product Unit

```lucid
// geometric_product.luc - Pipelined geometric product for Cl(2,0)

module geometric_product (
    input clk,
    input rst,
    input start,
    input a[multivector::mv2d],
    input b[multivector::mv2d],
    output result[multivector::mv2d],
    output done
  ) {
  
  // Multiplication table for Cl(2,0)
  // Each entry: (index_i, index_j) -> (sign, index_result)
  //
  // Geometric product: (a₀ + a₁e₁ + a₂e₂ + a₃e₁₂) · (b₀ + b₁e₁ + b₂e₂ + b₃e₁₂)
  //
  // Products:
  // 1·1 = 1         (0,0) -> (+1, 0)
  // 1·e₁ = e₁       (0,1) -> (+1, 1)
  // 1·e₂ = e₂       (0,2) -> (+1, 2)
  // 1·e₁₂ = e₁₂     (0,3) -> (+1, 3)
  // e₁·1 = e₁       (1,0) -> (+1, 1)
  // e₁·e₁ = 1       (1,1) -> (+1, 0)  // e₁² = 1
  // e₁·e₂ = e₁₂     (1,2) -> (+1, 3)
  // e₁·e₁₂ = e₂     (1,3) -> (+1, 2)
  // e₂·1 = e₂       (2,0) -> (+1, 2)
  // e₂·e₁ = -e₁₂    (2,1) -> (-1, 3)  // Anticommutative
  // e₂·e₂ = 1       (2,2) -> (+1, 0)  // e₂² = 1
  // e₂·e₁₂ = -e₁    (2,3) -> (-1, 1)
  // e₁₂·1 = e₁₂     (3,0) -> (+1, 3)
  // e₁₂·e₁ = -e₂    (3,1) -> (-1, 2)
  // e₁₂·e₂ = e₁     (3,2) -> (+1, 1)
  // e₁₂·e₁₂ = -1    (3,3) -> (-1, 0)  // Bivector squares to -1
  
  // Pipeline stages
  .clk(clk), .rst(rst) {
    dff stage[3];           // 3-stage pipeline
    dff result_reg[multivector::mv2d];
    dff done_reg;
  }
  
  // DSP48E1 multipliers (use FPGA hard multipliers)
  sig product[16][32];      // 16 products, each 32-bit
  sig scaled[16][16];       // Scaled back to Q8.8
  
  // Temporary accumulators for each component
  sig acc_scalar[32];
  sig acc_e1[32];
  sig acc_e2[32];
  sig acc_e12[32];
  
  always {
    done = done_reg.q;
    result = result_reg.q;
    
    // Pipeline control
    if (start) {
      stage.d = 3b001;
      done_reg.d = 0;
    } else if (stage.q != 0) {
      stage.d = stage.q << 1;
    }
    
    // Stage 1: Multiply all 16 component pairs
    if (stage.q[0]) {
      product[0] = $signed(a.scalar) * $signed(b.scalar);   // 1·1
      product[1] = $signed(a.scalar) * $signed(b.e1);       // 1·e₁
      product[2] = $signed(a.scalar) * $signed(b.e2);       // 1·e₂
      product[3] = $signed(a.scalar) * $signed(b.e12);      // 1·e₁₂
      product[4] = $signed(a.e1) * $signed(b.scalar);       // e₁·1
      product[5] = $signed(a.e1) * $signed(b.e1);           // e₁·e₁
      product[6] = $signed(a.e1) * $signed(b.e2);           // e₁·e₂
      product[7] = $signed(a.e1) * $signed(b.e12);          // e₁·e₁₂
      product[8] = $signed(a.e2) * $signed(b.scalar);       // e₂·1
      product[9] = $signed(a.e2) * $signed(b.e1);           // e₂·e₁
      product[10] = $signed(a.e2) * $signed(b.e2);          // e₂·e₂
      product[11] = $signed(a.e2) * $signed(b.e12);         // e₂·e₁₂
      product[12] = $signed(a.e12) * $signed(b.scalar);     // e₁₂·1
      product[13] = $signed(a.e12) * $signed(b.e1);         // e₁₂·e₁
      product[14] = $signed(a.e12) * $signed(b.e2);         // e₁₂·e₂
      product[15] = $signed(a.e12) * $signed(b.e12);        // e₁₂·e₁₂
    }
    
    // Stage 2: Scale products (Q8.8 * Q8.8 = Q16.16, shift right by 8)
    if (stage.q[1]) {
      scaled[0] = product[0][23:8];    // Keep middle 16 bits
      scaled[1] = product[1][23:8];
      scaled[2] = product[2][23:8];
      scaled[3] = product[3][23:8];
      scaled[4] = product[4][23:8];
      scaled[5] = product[5][23:8];
      scaled[6] = product[6][23:8];
      scaled[7] = product[7][23:8];
      scaled[8] = product[8][23:8];
      scaled[9] = product[9][23:8];
      scaled[10] = product[10][23:8];
      scaled[11] = product[11][23:8];
      scaled[12] = product[12][23:8];
      scaled[13] = product[13][23:8];
      scaled[14] = product[14][23:8];
      scaled[15] = product[15][23:8];
      
      // Accumulate according to multiplication table
      // Scalar component (index 0):
      acc_scalar = $signed(scaled[0])   // 1·1 -> scalar
                 + $signed(scaled[5])   // e₁·e₁ -> scalar
                 + $signed(scaled[10])  // e₂·e₂ -> scalar
                 - $signed(scaled[15]); // e₁₂·e₁₂ -> -scalar
      
      // e₁ component (index 1):
      acc_e1 = $signed(scaled[1])       // 1·e₁ -> e₁
             + $signed(scaled[4])       // e₁·1 -> e₁
             - $signed(scaled[11])      // e₂·e₁₂ -> -e₁
             + $signed(scaled[14]);     // e₁₂·e₂ -> e₁
      
      // e₂ component (index 2):
      acc_e2 = $signed(scaled[2])       // 1·e₂ -> e₂
             + $signed(scaled[7])       // e₁·e₁₂ -> e₂
             + $signed(scaled[8])       // e₂·1 -> e₂
             - $signed(scaled[13]);     // e₁₂·e₁ -> -e₂
      
      // e₁₂ component (index 3):
      acc_e12 = $signed(scaled[3])      // 1·e₁₂ -> e₁₂
              + $signed(scaled[6])      // e₁·e₂ -> e₁₂
              - $signed(scaled[9])      // e₂·e₁ -> -e₁₂
              + $signed(scaled[12]);    // e₁₂·1 -> e₁₂
    }
    
    // Stage 3: Write results
    if (stage.q[2]) {
      result_reg.d.scalar = acc_scalar[15:0];
      result_reg.d.e1 = acc_e1[15:0];
      result_reg.d.e2 = acc_e2[15:0];
      result_reg.d.e12 = acc_e12[15:0];
      done_reg.d = 1;
    }
  }
}
```

### 3. Register File

```lucid
// register_file.luc - Four multivector registers

module register_file (
    input clk,
    input rst,
    input write_enable,
    input write_addr[2],      // 2 bits = 4 registers
    input write_data[multivector::mv2d],
    input read_addr_a[2],
    input read_addr_b[2],
    output read_data_a[multivector::mv2d],
    output read_data_b[multivector::mv2d]
  ) {
  
  .clk(clk), .rst(rst) {
    // Four multivector registers
    dff r0[multivector::mv2d](#INIT(multivector::ZERO));
    dff r1[multivector::mv2d](#INIT(multivector::ZERO));
    dff r2[multivector::mv2d](#INIT(multivector::ZERO));
    dff r3[multivector::mv2d](#INIT(multivector::ZERO));
  }
  
  always {
    // Write
    if (write_enable) {
      case (write_addr) {
        2b00: r0.d = write_data;
        2b01: r1.d = write_data;
        2b10: r2.d = write_data;
        2b11: r3.d = write_data;
      }
    }
    
    // Read port A
    case (read_addr_a) {
      2b00: read_data_a = r0.q;
      2b01: read_data_a = r1.q;
      2b10: read_data_a = r2.q;
      2b11: read_data_a = r3.q;
      default: read_data_a = multivector::ZERO;
    }
    
    // Read port B
    case (read_addr_b) {
      2b00: read_data_b = r0.q;
      2b01: read_data_b = r1.q;
      2b10: read_data_b = r2.q;
      2b11: read_data_b = r3.q;
      default: read_data_b = multivector::ZERO;
    }
  }
}
```

### 4. Instruction Set & Decoder

```lucid
// geocpu_isa.luc - Instruction set for GeoCPU

module geocpu_isa {
  // Instruction encoding: 16 bits
  // [15:12] opcode
  // [11:10] dest register
  // [9:8]   src1 register
  // [7:6]   src2 register
  // [5:0]   immediate/flags
  
  // Opcodes
  const OP_NOP    = 4h0;   // No operation
  const OP_GP     = 4h1;   // Geometric product: dst = src1 · src2
  const OP_ADD    = 4h2;   // Addition: dst = src1 + src2
  const OP_SUB    = 4h3;   // Subtraction: dst = src1 - src2
  const OP_GRADE  = 4h4;   // Grade projection: dst = ⟨src1⟩_k
  const OP_LOAD   = 4h5;   // Load from memory: dst = mem[addr]
  const OP_STORE  = 4h6;   // Store to memory: mem[addr] = src1
  const OP_LOADI  = 4h7;   // Load immediate: dst = immediate multivector
  const OP_SCALAR = 4h8;   // Extract scalar: dst.scalar = src1.scalar
  const OP_BIVEC  = 4h9;   // Extract bivector: dst = ⟨src1⟩₂
  const OP_EMBED  = 4hA;   // Embed Boolean: dst = ι(pattern)
  const OP_MEASURE= 4hB;   // Measure: output scalar > threshold
  const OP_HALT   = 4hF;   // Halt execution
  
  struct instruction {
    opcode : 4;
    dst    : 2;
    src1   : 2;
    src2   : 2;
    imm    : 6;
  }
  
  // Decode instruction word
  function decode(inst_word[16]) {
    var inst instruction;
    inst.opcode = inst_word[15:12];
    inst.dst = inst_word[11:10];
    inst.src1 = inst_word[9:8];
    inst.src2 = inst_word[7:6];
    inst.imm = inst_word[5:0];
    return inst;
  }
}
```

### 5. Top-Level GeoCPU

```lucid
// geocpu_top.luc - Top-level GeoCPU module

module geocpu_top (
    input clk,              // 100MHz clock
    input rst_n,            // Active-low reset
    input rx,               // UART RX
    output tx,              // UART TX
    output led[8],          // Status LEDs
    input button            // Step button
  ) {
  
  .clk(clk) {
    // Reset conditioner
    reset_conditioner reset_cond;
    
    .rst(reset_cond.rst) {
      // Core components
      register_file reg_file;
      geometric_product gp_unit;
      
      // Program counter
      dff pc[8](#INIT(0));
      
      // Instruction register
      dff ir[16];
      
      // State machine
      fsm state = {FETCH, DECODE, EXECUTE, WRITEBACK, HALT};
      
      // Execution state
      dff exec_cycles[4];    // Counts cycles for multi-cycle ops
      
      // Button edge detector
      edge_detector button_edge(#RISE(1), #FALL(0));
      
      // LED display (show R0 scalar component)
      dff led_val[8];
    }
  }
  
  // Program ROM (simple programs for demo)
  program_rom prog_rom;
  
  // Temp signals
  sig inst[geocpu_isa::instruction];
  sig reg_a[multivector::mv2d];
  sig reg_b[multivector::mv2d];
  sig result[multivector::mv2d];
  
  always {
    // Reset
    reset_cond.in = ~rst_n;
    button_edge.in = button;
    
    // Default outputs
    tx = 1;  // UART idle
    led = led_val.q;
    
    // Register file connections
    reg_file.write_enable = 0;
    reg_file.write_addr = inst.dst;
    reg_file.write_data = result;
    reg_file.read_addr_a = inst.src1;
    reg_file.read_addr_b = inst.src2;
    reg_a = reg_file.read_data_a;
    reg_b = reg_file.read_data_b;
    
    // GP unit connections
    gp_unit.start = 0;
    gp_unit.a = reg_a;
    gp_unit.b = reg_b;
    
    // Instruction decode
    inst = geocpu_isa::decode(ir.q);
    
    // State machine
    case (state.q) {
      state.FETCH:
        // Fetch instruction from ROM
        ir.d = prog_rom.instruction;
        prog_rom.addr = pc.q;
        state.d = state.DECODE;
      
      state.DECODE:
        // Decode already happens combinatorially
        exec_cycles.d = 0;
        state.d = state.EXECUTE;
      
      state.EXECUTE:
        case (inst.opcode) {
          geocpu_isa::OP_NOP:
            state.d = state.WRITEBACK;
          
          geocpu_isa::OP_GP:
            // Geometric product (3 cycles)
            if (exec_cycles.q == 0) {
              gp_unit.start = 1;
              exec_cycles.d = 1;
            } else if (gp_unit.done) {
              result = gp_unit.result;
              state.d = state.WRITEBACK;
            } else {
              exec_cycles.d = exec_cycles.q + 1;
            }
          
          geocpu_isa::OP_ADD:
            // Component-wise addition
            result.scalar = $signed(reg_a.scalar) + $signed(reg_b.scalar);
            result.e1 = $signed(reg_a.e1) + $signed(reg_b.e1);
            result.e2 = $signed(reg_a.e2) + $signed(reg_b.e2);
            result.e12 = $signed(reg_a.e12) + $signed(reg_b.e12);
            state.d = state.WRITEBACK;
          
          geocpu_isa::OP_SUB:
            // Component-wise subtraction
            result.scalar = $signed(reg_a.scalar) - $signed(reg_b.scalar);
            result.e1 = $signed(reg_a.e1) - $signed(reg_b.e1);
            result.e2 = $signed(reg_a.e2) - $signed(reg_b.e2);
            result.e12 = $signed(reg_a.e12) - $signed(reg_b.e12);
            state.d = state.WRITEBACK;
          
          geocpu_isa::OP_SCALAR:
            // Extract scalar component
            result = multivector::ZERO;
            result.scalar = reg_a.scalar;
            state.d = state.WRITEBACK;
          
          geocpu_isa::OP_BIVEC:
            // Extract bivector (e₁₂) component
            result = multivector::ZERO;
            result.e12 = reg_a.e12;
            state.d = state.WRITEBACK;
          
          geocpu_isa::OP_LOADI:
            // Load immediate value into scalar
            result = multivector::ZERO;
            result.scalar = inst.imm << 2;  // Scale up
            state.d = state.WRITEBACK;
          
          geocpu_isa::OP_HALT:
            state.d = state.HALT;
          
          default:
            state.d = state.WRITEBACK;
        }
      
      state.WRITEBACK:
        // Write result to register
        if (inst.opcode != geocpu_isa::OP_NOP && 
            inst.opcode != geocpu_isa::OP_HALT) {
          reg_file.write_enable = 1;
        }
        
        // Increment PC
        pc.d = pc.q + 1;
        
        // Update LED display (show R0.scalar as 8-bit value)
        led_val.d = reg_file.read_data_a.scalar[15:8];
        
        // Next instruction
        state.d = state.FETCH;
      
      state.HALT:
        // Stay halted until reset
        // Flash LEDs to show halt state
        if (exec_cycles.q[23]) {
          led_val.d = 8hFF;
        } else {
          led_val.d = 8h00;
        }
        exec_cycles.d = exec_cycles.q + 1;
    }
  }
}
```

### 6. Example Program ROM

```lucid
// program_rom.luc - Sample programs

module program_rom (
    input addr[8],
    output instruction[16]
  ) {
  
  // Example program: Compute P₁ AND P₂ using geometric product
  // 
  // Program:
  //   R0 = ι(P₁)        // Load P₁ (0.5 + 0.5·e₁)
  //   R1 = ι(P₂)        // Load P₂ (0.5 + 0.5·e₂)
  //   R2 = R0 · R1      // Geometric product
  //   R3 = ⟨R2⟩₀        // Extract scalar (truth value)
  //   HALT
  
  const PROGRAM = {
    // Instruction 0: LOADI R0, P1_scalar (0.5 = 0x80 in Q8.8)
    16b0111_00_00_00_100000,  // LOADI R0, 0x20 (will be shifted left 2)
    
    // Instruction 1: Load P1 e1 component
    // Actually, let's hardcode this in a load-from-pattern instruction
    // For now, use explicit loads
    
    // Simpler demo: R0 = 1.0, R1 = 1.0, R2 = R0·R1
    16b0111_00_00_00_000001,  // LOADI R0, 1 (scalar = 1.0)
    16b0111_01_00_00_000001,  // LOADI R1, 1 (scalar = 1.0)
    16b0001_10_00_01_000000,  // GP R2, R0, R1
    16b1111_00_00_00_000000,  // HALT
    
    // Pad remaining with NOPs
    16b0000_00_00_00_000000,  // NOP
    16b0000_00_00_00_000000,
    16b0000_00_00_00_000000
    // ... (total 256 instructions)
  };
  
  always {
    instruction = PROGRAM[addr];
  }
}
```

## Constraints File (Alchitry Au V2)

```tcl
# alchitry_au_v2.acf - Pin constraints for GeoCPU

# Clock
set_property PACKAGE_PIN N14 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports clk]

# Reset button
set_property PACKAGE_PIN P6 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

# LEDs
set_property PACKAGE_PIN K13 [get_ports {led[0]}]
set_property PACKAGE_PIN K12 [get_ports {led[1]}]
set_property PACKAGE_PIN L14 [get_ports {led[2]}]
set_property PACKAGE_PIN L13 [get_ports {led[3]}]
set_property PACKAGE_PIN M16 [get_ports {led[4]}]
set_property PACKAGE_PIN M14 [get_ports {led[5]}]
set_property PACKAGE_PIN M12 [get_ports {led[6]}]
set_property PACKAGE_PIN N16 [get_ports {led[7]}]
set_property IOSTANDARD LVCMOS33 [get_ports led*]

# Button
set_property PACKAGE_PIN P5 [get_ports button]
set_property IOSTANDARD LVCMOS33 [get_ports button]

# UART
set_property PACKAGE_PIN A11 [get_ports rx]
set_property PACKAGE_PIN A10 [get_ports tx]
set_property IOSTANDARD LVCMOS33 [get_ports {rx tx}]
```

## Resource Estimates

**For n=2 (4D Clifford algebra):**

| Component | LUTs | FFs | DSPs | BRAM |
|-----------|------|-----|------|------|
| Register File (4×4×16-bit) | ~200 | 256 | 0 | 0 |
| Geometric Product Unit | ~800 | ~500 | 16 | 0 |
| Grade Projection | ~100 | ~50 | 0 | 0 |
| Control Logic | ~300 | ~200 | 0 | 0 |
| Program ROM | ~50 | 0 | 0 | 1 |
| **Total** | **~1500** | **~1000** | **16** | **1** |

**Utilization: ~7% of XC7A35T** ✓ Very comfortable!

## Performance Estimates

- **Clock**: 100MHz (10ns period)
- **Geometric Product**: 3 cycles = 30ns
- **Simple ops** (add/sub/project): 1 cycle = 10ns
- **Throughput**: ~33M geometric products/second

**Comparison to software:**
- Python/NumPy: ~1M ops/second
- **Speedup: ~33×**

## Demo Programs

### Program 1: Boolean AND via Geometric Product

```lucid
// Encode: P₁ = 0.5 + 0.5·e₁, P₂ = 0.5 + 0.5·e₂
// Compute: P₁ · P₂ = 0.25 + 0.25·e₁ + 0.25·e₂ + 0.25·e₁₂
// Extract scalar: 0.25 (25% probability)

R0 = {0.5, 0.5, 0, 0}     // P₁
R1 = {0.5, 0, 0.5, 0}     // P₂  
R2 = R0 · R1              // GP
R3 = scalar(R2)           // Extract → 0.25
```

### Program 2: Correlation Detection

```lucid
// Detect if two sensors are correlated
// Load sensor data as multivectors
// Check bivector component

R0 = sensor_A_mv          // Load sensor A pattern
R1 = sensor_B_mv          // Load sensor B pattern
R2 = R0 · R1              // Geometric product
R3 = bivector(R2)         // Extract e₁₂ component
if (|R3.e12| > threshold) {
  CORRELATED = true
}
```

## Next Steps

1. **Synthesize the basic design** - Get it running on hardware
2. **Add UART interface** - Upload programs from PC
3. **Implement Boolean embedding** - Precompute embeddings, store in ROM
4. **Performance testing** - Benchmark against software
5. **Scale to n=3** - 8D Clifford algebra (64 products!)
