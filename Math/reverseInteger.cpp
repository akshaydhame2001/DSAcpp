/*
Given a signed 32-bit integer x, return x with its digits reversed.
If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.
Also trailing 0 shoud not appear like for 10400 it should be 401 and not 00401.
*/
#include <bits/stdc++.h>
using namespace std;

int reverse(int x)
{
    int ans = 0;
    while (x != 0)
    {
        int digit = x % 10;
        if ((ans > INT_MAX / 10) || (ans < INT_MIN / 10))
        {
            return 0;
        }
        ans = ans * 10 + digit;
        x = x / 10;
    }
    return ans;
}

int main()
{
    int n;
    cin >> n;
    cout << reverse(n) << endl;
    return 0;
}