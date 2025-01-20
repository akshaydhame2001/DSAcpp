/*
Given an integer x, return true if x is a palindrome, and false otherwise.
-231 <= x <= 231 - 1
Follow up: Could you solve it without converting the integer to a string?

*/

#include <bits/stdc++.h>
using namespace std;

bool isPalindrome(int x)
{
    if (x < 0)
    {
        return false;
    }
    int reverse = 0;
    int num = x;
    while (num > 0)
    {
        int d = num % 10;
        if ((reverse > INT_MAX / 10) || (reverse < INT_MIN / 10))
        {
            return false;
        }
        reverse = reverse * 10 + d;
        num = num / 10;
    }
    return reverse == x;
}

int main()
{
    int n;
    cin >> n;
    cout << (isPalindrome(n) ? "true" : "false") << endl;
    return 0;
}