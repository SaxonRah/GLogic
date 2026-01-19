from dataclasses import dataclass

# Row order: TT, TF, FT, FF  (bits 3..0)
TT, TF, FT, FF = 3, 2, 1, 0

def bits4(m: int) -> tuple[int, int, int, int]:
    """Return outputs as (TT, TF, FT, FF) bits."""
    m &= 0xF
    return ((m >> 3) & 1, (m >> 2) & 1, (m >> 1) & 1, (m >> 0) & 1)

def show(m: int) -> str:
    a,b,c,d = bits4(m)
    return f"{a},{b},{c},{d}  (0b{m&0xF:04b}=0x{m&0xF:X})"

# Base functions as masks
F = 0x0         # 0000  FALSE
T = 0xF         # 1111  TRUE
P = 0xC         # 1100  p
Q = 0xA         # 1010  q

def NOT(x: int) -> int: return (~x) & 0xF
def AND(x: int, y: int) -> int: return (x & y) & 0xF
def OR(x: int, y: int) -> int: return (x | y) & 0xF
def XOR(x: int, y: int) -> int: return (x ^ y) & 0xF
def XNOR(x: int, y: int) -> int: return NOT(XOR(x, y))
def IMP(x: int, y: int) -> int: return OR(NOT(x), y)   # x -> y
def NAND(x: int, y: int) -> int: return NOT(AND(x, y))
def NOR(x: int, y: int) -> int: return NOT(OR(x, y))

# Quick demo: a sequence of bitmask steps that yields all-ones or all-zeros
if __name__ == "__main__":
    print("P:", show(P))
    print("Q:", show(Q))

    # Example: (p OR not p) -> TRUE
    expr_true = OR(P, NOT(P))
    print("p OR ~p:", show(expr_true))

    # Example: (p AND not p) -> FALSE
    expr_false = AND(P, NOT(P))
    print("p AND ~p:", show(expr_false))

    # Example: XOR then XNOR yields TRUE (since XNOR(x,x)=TRUE)
    x = XOR(P, Q)
    expr_true2 = XNOR(x, x)
    print("x = p XOR q:", show(x))
    print("XNOR(x,x):", show(expr_true2))
