I am refactoring the code for better understanding and easier optimization.   
I have extracted "calc_func" to "multicalc.1.cu", then added some helper funcions. 

However, I found some bugs preventing me forward.  
In file "multicalc.cu" there are several comments describing bugs I found and how to fix them.  
This file is based on the ORIGINAL one, so it should be easy to just diff them.  
If you agree with me, please apply the fix and test it again on your system.  

When fixed, the function should be functionally equivalent to that in "multicalc.1.cu".   
No performance gains yet, but you should test it as well,
 to make sure I'm not making mistakes. 

The new code explains itself, i.e., the logic of this code is easier to follow and understand.  
The final version will be based on it. 

Fluorine Dog  
From Linux without IME
