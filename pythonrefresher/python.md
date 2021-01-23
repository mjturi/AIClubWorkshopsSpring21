---
layout: page
title: Python Refresher
show_in_menu: false
disable_anchors: true
---

# test
```python
hello = 5
```

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
Python has six standard data types. In this portion we will only look through 3 of the data types; Integer, Float, and boolean. We can use the ```type()``` function to check the type of data type in a variable.
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
10.5 <class 'float'>
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
