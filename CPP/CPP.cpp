#include <bits/stdc++.h>
using namespace std;

// data types: int, char, float, double, bool

int main()
{
    int x = 10;
    int *ptr = &x; // Pointer stores the address of x

    cout << "Value of x: " << x << endl;
    cout << "Address of x: " << &x << endl;
    cout << "Pointer ptr points to address: " << ptr << endl;
    cout << "Value at pointer address: " << *ptr << endl;

    return 0;

    /*
    Value of x: 10
    Address of x: 0x61ff08
    Pointer ptr points to address: 0x61ff08
    Value at pointer address: 10

    */
}

// arrays are passed by reference