#include <bits/stdc++.h>
using namespace std;

void printRecursion(int i, int n)
{
    if (i > n)
        return;
    cout << i << endl;
    printRecursion(i + 1, n);
}
void printBacktracking(int i, int n)
{
    if (i < 1)
        return;
    printBacktracking(i - 1, n);
    cout << i << endl;
}
void printNos(int n)
{
    // Your code here
    if (n == 0)
        return;
    printNos(n - 1);
    cout << n << " ";
}
int printSum(int n)
{
    // Your code here
    if (n == 0)
        return 0;
    return n + printSum(n - 1);
}
int fact(int n)
{
    if (n == 1)
        return 1;
    return n * fact(n - 1);
}

int main()
{
    // Write C++ code here
    int n;
    cin >> n;
    printRecursion(1, n);
    printBacktracking(n, n);
    printNos(n);
    cout << endl;
    cout << printSum(n) << endl;
    cout << fact(n);

    return 0;
}