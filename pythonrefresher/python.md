---
layout: page
title: Python Refresher
show_in_menu: false
disable_anchors: true
---

## Introduction
### Syntax
Let's remember the syntax of the python language. We don't have to declare a specific data type or use semicolons
```python
# Comments are specified with a pound sign
name = "Hello" # Here the variable name stores the string hello
```

### Print Statements
To print to the console, we use ```print()``` function in Python
```python
print("Hello World") #Simple print statement

name = "Hector"
print(name)
```
**output**
```
Hello World
Hector
```

### Data Types
Python has six standard data types. In this portion we will only look through 3 of the data types; Integer, Float, and boolean. We can use the ```type()``` function to check the type of a variable.
```python
number = 100 # This is an integer because it is a whole number
decimal = 10.5 # This is a float because it has a decimal
boolean = True # Boolean variables contain either a 'True' or 'False'

print(number, type(number))
print(decimal, type(decimal))
print(boolean, type(boolean))
```
**output**
```
100 <class 'int'>
10.5 <class 'float'>
True <class 'bool'>
```

### List
A list in python is a collection of variables denoted by square brackets. Unlike other languages, we can store different data types in a single list

```python
example_list = [1, 5 , 5 , 9] #This is a list of integers
different_data_types = [1 , 20.5, "Hello World", False] #This list has different data types within and it still works

print(example_list, "is of type", type(example_list))
print(different_data_types, "is of type", type(example_list))
```
**output**
```
[1, 5, 5, 9] is of type <class 'list'>
[1, 20.5, 'Hello World', False] is of type <class 'list'>
```

### If Statements
If statements evaluate logical conditions and only execute if the condition is true
```python
number_one = 100
number_two = 150

if number_one < number_two:
    print("SDSU AI Club")

if number_one % 2 == 0:
    print(100/2)

if 3 % 2 == 1:
    print(3/2)
```
**output**
```
SDSU AI Club
50.0
1.5
```

## Advanced
### For Loops
Python for loops are used when we know how many times we'll iterate through something
```python
for x in range(5): #Here x is the iterater in the range 0-4
    print(x)
    
random_list = [1,2,3,7,8,9]
for num in random_list:
  print(num)
```

### While Loops
While loops are used whenever we don't know how many times we'll be iterating through something
```python
a = 0
while a != 5:
  a = a + 1
```

### Importing Modules
When importing modules, we usually keep this information at the top of our script
```python
#To use math functions we must import the math module
import math

x = math.pi #Here we are calling the math module to use its pi function
print(x)
```

We can also give an alias to our modules for future reference
```python
#We can also give aliases to packages
import math as m #You can change this alias to what you want

x = m.pi #We're doing the same as the code above except we call math with an alias 'm'
print(x)
```
