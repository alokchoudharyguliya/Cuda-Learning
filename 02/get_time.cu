#include <windows.h>
double get_time(){
    LARGE_INTEGER freq, cntr;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cntr);
    return (double)cntr.QuadPart/freq.QuadPart;
}