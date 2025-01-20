/*
You are given a 3-digit number n, Find whether it is an Armstrong number or not.
An Armstrong number of three digits is a number such that the sum of the cubes of its digits is equal to
the number itself. 371 is an Armstrong number since 33 + 73 + 13 = 371.
Note: Return true if it is an Armstrong number else return false.
*/
#include <bits/stdc++.h>
using namespace std;

bool armstrongNumber(int n)
{
    // code here
    int num = n;
    int sum = 0;
    while (num > 0)
    {
        int d = num % 10;
        sum += d * d * d;
        num = num / 10;
    }
    return sum == n;
}

int main()
{
    int n;
    cin >> n;
    cout << (armstrongNumber(n) ? "true" : "false") << endl;
    return 0;
}
