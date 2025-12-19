# Mojo Quick Reference

**Version:** Mojo 0.26.1 (nightly)
**Last Updated:** 2025-12-13

---

## Core Fundamentals

### fn vs def

| Feature       | `fn`                   | `def`                       |
| ------------- | ---------------------- | --------------------------- |
| Type checking | Strict (enforced)      | Flexible (hints only)       |
| Can raise     | Must add `raises`      | Raises by default           |
| Performance   | Compiled, fast         | Python-like                 |
| Use when      | Hot paths, strict code | Prototyping, Python interop |

```mojo
fn strict_add(x: Int, y: Int) -> Int:
    return x + y  # Types enforced, cannot raise

def flexible_add(x, y):
    return x + y  # Types optional, can raise
```

### Variables

| Keyword | Scope        | Mutability | Use               |
| ------- | ------------ | ---------- | ----------------- |
| `var`   | Lexical      | Mutable    | Normal variables  |
| `alias` | Compile-time | Immutable  | Constants, types  |
| `let`   | Deprecated   | -          | Use `var` instead |

```mojo
var x = 10              # Runtime mutable
var y: Int              # Declare without init
alias PI = 3.14159      # Compile-time constant

# Lexical scoping (var)
var num = 1
if True:
    var num = 2  # New inner scope variable
    print(num)   # 2
print(num)       # 1
```

### Basic Types

```mojo
var i: Int = 42
var f: Float64 = 3.14
var s: String = "hello"
var b: Bool = True

# Type aliases (scalars are SIMD[DType, 1])
var i8: Int8 = 127
var i32: Int32 = 2147483647
var u32: UInt32 = 4294967295
var f32: Float32 = 3.14
```

---

## Control Flow

### If-Elif-Else

```mojo
if x < 0:
    print("negative")
elif x == 0:
    print("zero")
else:
    print("positive")
```

### For Loops

```mojo
# Range iteration
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# Collection iteration
for item in my_list:
    print(item[])

# Break and continue
for i in range(10):
    if i == 3:
        continue  # Skip 3
    if i == 7:
        break     # Stop at 7
    print(i)

# Else clause (runs if no break)
for i in range(5):
    if i == 10:
        break
else:
    print("No break occurred")
```

### While Loops

```mojo
var n = 0
while n < 5:
    print(n)
    n += 1

# Else clause (runs if condition becomes false naturally)
while n < 10:
    n += 1
    if n == 7:
        break
else:
    print("Completed without break")
```

---

## Functions

### Basic Function

```mojo
fn greet(name: String) -> String:
    return "Hello, " + name

def add(a: Int, b: Int) -> Int:
    return a + b
```

### Default Arguments

```mojo
fn power(base: Int, exp: Int = 2) -> Int:
    return base ** exp

power(3)      # 9 (uses default exp=2)
power(3, 3)   # 27
```

### Keyword Arguments

```mojo
fn my_pow(base: Int, exp: Int = 2) -> Int:
    return base ** exp

my_pow(exp=3, base=2)  # Order doesn't matter
```

### Variadic Arguments

```mojo
fn sum(*values: Int) -> Int:
    var total: Int = 0
    for value in values:
        total += value
    return total

sum(1, 2, 3, 4)  # 10
```

### Positional/Keyword-Only

```mojo
# Args before '/' are positional-only
# Args after '*' are keyword-only
def func(pos_only, /, pos_or_kw, *, kw_only):
    pass

func(1, 2, kw_only=3)  # Valid
```

---

## Error Handling

### Raising Errors

```mojo
def risky_func(n: Int) -> Int:
    if n < 0:
        raise Error("Negative not allowed")
    return n * 2

# String literal shorthand
def check(n: Int):
    if n > 100:
        raise "Value too large"
```

### Try-Except

```mojo
try:
    result = risky_func(-5)
except e:
    print("Error:", e)
else:
    print("Success:", result)  # Runs if no error
finally:
    print("Cleanup")  # Always runs
```

### Propagating Errors

```mojo
# fn cannot raise by default - must add 'raises'
fn call_risky() raises:
    risky_func(10)  # Propagates error up

# Handle in fn
fn safe_call():
    try:
        risky_func(10)
    except e:
        print("Handled:", e)
```

---

## Structs & Traits

### Basic Struct

```mojo
@value  # Auto-generates __init__, __copyinit__, __moveinit__
struct Point:
    var x: Float64
    var y: Float64

var p = Point(3.0, 4.0)
print(p.x, p.y)
```

### Struct with Methods

```mojo
struct Rectangle:
    var width: Float64
    var height: Float64

    fn __init__(out self, width: Float64, height: Float64):
        self.width = width
        self.height = height

    fn area(self) -> Float64:
        return self.width * self.height

var rect = Rectangle(10.0, 5.0)
print(rect.area())  # 50.0
```

### Traits

```mojo
trait Quackable:
    fn quack(self): ...

struct Duck(Quackable):
    fn quack(self):
        print("Quack!")

struct Cow(Quackable):
    fn quack(self):
        print("Moo!")  # Cows can quack too!
```

### Common Traits

| Trait        | Purpose           | Required Method       |
| ------------ | ----------------- | --------------------- |
| `Stringable` | Convert to String | `__str__() -> String` |
| `Boolable`   | Boolean context   | `__bool__() -> Bool`  |
| `Movable`    | Move semantics    | `__moveinit__()`      |
| `Copyable`   | Copy semantics    | `__copyinit__()`      |

```mojo
struct Foo(Stringable):
    var value: Int

    fn __str__(self) -> String:
        return "Foo(" + String(self.value) + ")"

print(Foo(42))  # Foo(42)
```

---

## Collections

### List

```mojo
from collections import List

# Create
var nums = List[Int]()
var names = List[String]("Alice", "Bob", "Charlie")

# Operations
nums.append(10)
nums.append(20)
print(nums[0])           # 10
print(len(nums))         # 2

# Iteration
for name in names:
    print(name[])

# Boolean context
if nums:
    print("List has items")
```

### Dict

```mojo
from collections import Dict

# Create
var scores = Dict[String, Int]()
scores["Alice"] = 95
scores["Bob"] = 87

# Check key
if "Alice" in scores:
    print(scores["Alice"])

# Iterate
for item in scores.items():
    print(item[].key, "->", item[].value)

# From literals
var colors = {"red": 255, "green": 128, "blue": 64}
```

### List Literals

```mojo
var list: List[Int] = [1, 2, 3]
var vec: SIMD[DType.uint8, 8] = [1, 2, 3, 4, 5, 6, 7, 8]
```

---

## Strings

### Creation & Concatenation

```mojo
var s1 = "hello"
var s2 = " world"
var s3 = s1 + s2  # "hello world"

# Efficient multi-arg constructor
var msg = String("Point: (", x, ", ", y, ")")
```

### Indexing & Slicing

```mojo
var text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
print(text[0])      # A
print(text[-1])     # Z
print(text[1:4])    # BCD
print(text[:6])     # ABCDEF
print(text[-6:])    # UVWXYZ
print(text[::2])    # ACEGIKMOQSUWY
print(text[::-1])   # Reverse
```

### Operations

```mojo
var s = "hello"

# Case conversion
s.upper()  # "HELLO"
s.lower()  # "hello"

# Justification
s.rjust(10, "*")  # "*****hello"
s.ljust(10, "-")  # "hello-----"
s.center(10, " ") # "  hello   "

# Replication
var repeated = s * 3  # "hellohellohello"

# Substring check
if "ell" in s:
    print("Found")

# Comparison (lexicographical)
"apple" < "banana"  # True
```

---

## Memory Ownership

### Parameter Conventions

| Convention | Symbol  | Meaning              | Use              |
| ---------- | ------- | -------------------- | ---------------- |
| Borrowed   | `self`  | Read-only reference  | Default, no copy |
| Mutable    | `mut`   | Mutable reference    | Modify in-place  |
| Owned      | `owned` | Takes ownership      | Consumes value   |
| Out        | `out`   | Uninitialized output | Constructors     |

```mojo
fn read_only(value: String):
    print(value)  # Can read, not modify

fn modify(mut buffer: List[Int]):
    buffer.append(42)  # Modifies original

fn consume(owned data: String):
    print(data)  # Takes ownership, data freed after

fn __init__(out self, value: Int):
    self.value = value  # Initialize uninitialized self
```

### Transfer Operator `^`

```mojo
var message = "hello"
consume(message^)  # Transfer ownership
# message is now invalid
```

---

## SIMD Operations

### Creating SIMD Vectors

```mojo
# Create 4-wide float32 vector
var vec1 = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

# Broadcast single value to all elements
var vec2 = SIMD[DType.float32, 4](2.0)

# Integer vectors
var int_vec = SIMD[DType.int32, 8](1, 2, 3, 4, 5, 6, 7, 8)
```

### Operations

```mojo
# Element-wise arithmetic
var sum = vec1 + vec2
var product = vec1 * vec2
var diff = vec1 - vec2

# Scalar operations
var scaled = vec1 * 2.0

# Fused multiply-add (a * b + c)
var result = vec1.fma(vec2, vec3)

# Reduction operations
var total = vec1.reduce_add()    # Sum all elements
var maximum = vec1.reduce_max()  # Max element
```

### Type Aliases

```mojo
# Scalars are SIMD[DType, 1]
var x: Float32 = 3.14  # SIMD[DType.float32, 1]
var y: Int64 = 100     # SIMD[DType.int64, 1]
```

---

## Python Interop

### Import Python Modules

```mojo
from python import Python

def main():
    # Import module
    var np = Python.import_module("numpy")

    # Use Python objects
    var array = np.array([1, 2, 3, 4, 5])
    var mean = np.mean(array)
    print(mean)

    # Python lists
    var py_list = Python.list()
    py_list.append("Mojo")
    py_list.append("Python")

    # Python dicts
    var py_dict = Python.dict()
    py_dict["key"] = "value"
```

### Add Python Path

```mojo
Python.add_to_path("path/to/modules")
var my_module = Python.import_module("my_module")
```

### Common Imports

```mojo
var math = Python.import_module("math")
var sys = Python.import_module("sys")
var pd = Python.import_module("pandas")
var builtins = Python.import_module("builtins")
```

---

## Parameters (Compile-Time)

### Parametric Functions

blo

```mojo
fn repeat[count: Int](msg: String):
    @parameter
    for i in range(count):
        print(msg)

repeat[3]("Hello")  # Prints "Hello" 3 times
```

### @parameter If

```mojo
@parameter
if debug_mode:
    print("Debug enabled")  # Only compiled if debug_mode is True
else:
    print("Production")     # Only compiled otherwise
```

### @parameter For (Loop Unrolling)

```mojo
@parameter
for i in range(4):
    print(i)  # Loop unrolled at compile-time
```

### Infer-Only Parameters

```mojo
# Parameters before '//' are inferred
fn process[T: Stringable, //, count: Int](msg: T):
    @parameter
    for i in range(count):
        print(String(msg))

process[3]("hello")  # T inferred as StringLiteral
```

---

## Advanced Features

### Generic Functions with Traits

```mojo
fn print_item[T: Stringable](item: T):
    print(String(item))

print_item(42)
print_item("hello")
```

### SIMD Width

```mojo
from sys.info import simdwidthof

fn process_simd[dtype: DType]():
    alias width = simdwidthof[dtype]()
    var vec = SIMD[dtype, width](1.0)
    print("SIMD width:", width)
```

### Unsafe Pointers (Advanced)

```mojo
from memory import UnsafePointer

var ptr = UnsafePointer[Int].alloc(10)
ptr[0] = 42
print(ptr[0])
ptr.free()
```

---

## Common Patterns

### Error Handling in fn

```mojo
fn safe_divide(a: Int, b: Int) -> Float64:
    try:
        if b == 0:
            raise "Division by zero"
        return Float64(a) / Float64(b)
    except e:
        print("Error:", e)
        return 0.0
```

### Builder Pattern

```mojo
struct Config:
    var name: String
    var timeout: Int

    fn __init__(out self):
        self.name = ""
        self.timeout = 30

    fn with_name(mut self, name: String) -> Self:
        self.name = name
        return self

    fn with_timeout(mut self, timeout: Int) -> Self:
        self.timeout = timeout
        return self

var config = Config().with_name("app").with_timeout(60)
```

### Optional Values

```mojo
from collections import Optional

var maybe: Optional[Int] = 42
if maybe:
    print("Value:", maybe.value())
```

---

## Quick Gotchas

- `def` functions raise by default, `fn` needs `raises` keyword
- `var` uses lexical scoping, implicit vars use function scoping
- `let` is deprecated, use `var`
- Uppercase letters < lowercase in string comparison
- SIMD operations require matching widths and dtypes
- Transfer ownership with `^` operator
- Traits must be implemented completely
- `@parameter` requires compile-time constants
- Python objects managed by Python GC

---

## Useful Imports

```mojo
from collections import List, Dict, Optional
from python import Python
from memory import UnsafePointer
from sys.info import simdwidthof
from testing import assert_raises
```

---

**Sources:**

- [Mojo Manual](https://docs.modular.com/mojo/manual/)
- [Mojo Standard Library](https://docs.modular.com/mojo/stdlib/)
- [Mojo Changelog](https://docs.modular.com/mojo/changelog/)
