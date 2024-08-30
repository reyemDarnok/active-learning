from time import process_time
from timeit import Timer
from subprocess import run

t = Timer(stmt=r'run([r"C:\FOCUS_PELMO.664\PELMO500.exe"])', timer=process_time, globals=globals())
cpu_time = t.timeit(number=20) / 20
t = Timer(stmt=r'run([r"C:\FOCUS_PELMO.664\PELMO500.exe"])', globals=globals())
wall_time = t.timeit(number=20) / 20
print(cpu_time)
print(wall_time)
