---
title: Solving A File Descriptor Challenge
date: 2024-01-17 09:25:00 +0800
categories: [ctf_writeup, pwn]
tags: [ctf_writeup]
---

# Intro
Hi! Today we're going to solve a challenge from the awesome website pwnable.kr that has to do with file descriptors. Without further ado, let's jump in.
# The Challenge
So the challenge runs on an ssh box. We log in as a user named `fd`, which stands for file descriptor, which is our main topic today, and in our home directory we find the following files:
![the-files](/assets/img/fd/Pasted image 20240116190427.png)
_The Files_
The interesting files here are `fd`, which is a binary, `fd.c`, which we presume to be the source code for the binary, and `flag`, which we don't have read permissions for. Let's look at the source code for `fd.c`:
![source-code](/assets/img/fd/Pasted image 20240116190546.png)
_The Source Code_
Alright. So we start by checking the number of arguments. Then, we define a new variable `fd`, which we define to be `atoi(argv[1]) - 0x1234`. The `atoi(3)` function takes a string as input and returns the corresponding number. If we were to just pass `argv[1]` without the call to `atoi()`, it wouldn't make sense because we would try to subtract `0x1234` from a string. After this, we define a new variable `len` and set it to `read(fd, buf, 32)`. Then we compare buf to the string `"LETMEWIN\n"` , and if they're equal, we cat the flag. The meat of the challenge is in the call to `read`. From the manpage for read:
![read-manpage](/assets/img/fd/Pasted image 20240116191022.png)
_The Read Manpage_
![read-manpage-return](/assets/img/fd/Pasted image 20240116191039.png)
_Return Value_
Okay, so let's unpack this a bit. The function `read()` takes three arguments: A `count`, a file descriptor (Like the name of the challenge), and a buffer. It reads `count` bytes from the file descriptor into the buffer.
# A file descriptor?
You may have heard the saying that ["In Linux, Everything is a file"](https://en.wikipedia.org/wiki/Everything_is_a_file) . For example, in Windows, Registry Keys are opened using the `RegOpenKey` family of functions, and files are opened using the `OpenFile` family of functions. Registry keys are treated as different objects than files. We can see this with the following Windows program I wrote:
```
#include <Windows.h>  
#include <stdio.h>  
  
int main() {  
HKEY hKey;  
int res = RegOpenKeyEx(HKEY_LOCAL_MACHINE, TEXT("SOFTWARE"), 0, KEY_READ, &hKey);  
HANDLE theFile = CreateFileA("D:\\the_file.txt", GENERIC_READ, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);  
  
while (getc(stdin) != 'a');  
  
printf("Closing Handles\n");  
  
RegCloseKey(hKey);  
CloseHandle(theFile);  
  
printf("Handles closed successfuly.\n");  

}
```
This program opens the `HKLM\SOFTWARE` registry key, and the `D:\the_file.txt`, and then closes the handles when it gets input. If we look at its open handles with the Sysinternals Process Explorer tool, before we close the handles, we see:
![open-handles](/assets/img/fd/Pasted image 20240116202044.png)
_Open Handles_
Which are the exact handles we opened! Cool, huh? But as we mentioned earlier, in Linux, things are different. In Linux, everything is a file (For example sockets and devices are also trated as files), and files are opened using the `open()` function. From the manpage for `open()`, we see that `open()` returns what's known as a file descriptor:
![open-manpage](/assets/img/fd/Pasted image 20240116210230.png)
_Open Manpage_
Let's look at an example. Consider the following program:
![open-program](/assets/img/fd/Pasted image 20240116210615.png)
_Open Program_
This program just creates a file names `file.txt` in the current directory, and prints the file descriptor associated with it. Upon running it, we see
`fd=3`
So a file descriptor is just an identifier associated with a file. If we were to open another file, with the following program, we'd get another file descriptor:
![two-fd](/assets/img/fd/Pasted image 20240116210914.png)
_Two File Descriptors_
`fd=3 fd2=4`
You may be wondering: If we just opened a file, why is the associated file descriptor `3`, and not something more logical, like `0` or `1`? That's a great question, and it's the key to solving this challenge. In addition to the regular file descriptors we just learned about, there are also special file descriptors; The file descriptor `0` is associated with `stdin`, the standard input stream, the file descriptor `1` is associated with `stdout`, the standard output stream, and finally, the file descriptor `2` is associated with `stderr`, the standard error stream. When you call a function that asks for input like `scanf()`, under the hood the operating system `read()`s from `stdin`, which corresponds to file descriptor `0`, meaning a call like this gets called: `read(0, ...)`. And when you print output, for example using `printf`, what you're actually doing under the hood is writing to the file `/dev/stdout`, which is the standard output stream. Cool, huh? 
Now, let's go back to our challenge. 
# A second look at the challenge
This is the code:
![second-code](/assets/img/fd/Pasted image 20240116190546.png)
_The Code Again_
So the `fd` is equal to our first argument minus `0x1234`. Then, we read into `buf` from `fd`. We want `buf` to be equal to `LETMEWIN\n`. So we need to pass input to `buf` somehow. We just learned that input is managed using file descriptor `0`! So we just need `fd` to be equal to `0`. Solving the equation `argv[1] - 0x1234 = 0` yields that `argv[1] = 0x1234`, or in decimal `4660`. Indeed, if we pass the argument `4660`, we can pass input!
![solved](/assets/img/fd/Pasted image 20240116211639.png)
_Solving The Challenge_
Nice!! We got the flag, but more importantly, we learned what file descriptors are. 

Hope you enjoyed this blogpost

Yoray
