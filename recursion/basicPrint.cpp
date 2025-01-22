#include <bits/stdc++.h>
using namespace std;

void print(int i, int n)
{
    if (i > n)
        return;
    cout << "Akshay" << endl;
    print(i + 1, n);
}

int main()
{
    // Write C++ code here
    int n;
    cin >> n;
    print(1, n);

    return 0;
}