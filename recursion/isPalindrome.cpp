#include <bits/stdc++.h>
using namespace std;

bool recursion(vector<char> &s, int i)
{
    int n = s.size();
    if (i >= n / 2)
    {
        return true;
    }
    if (s[i] != s[n - i - 1])
    {
        return false;
    }
    return recursion(s, i + 1);
}

bool isPalindrome(string s)
{
    vector<char> str; // Use a vector because strings in C++ are immutable
    for (auto &i : s)
    {
        if (isalnum(i))
        {
            str.push_back(tolower(i));
        }
    }
    return recursion(str, 0);
}

int main()
{
    string s;
    cout << "Enter String: ";
    cin >> s;
    cout << "isPalindrome: " << (isPalindrome(s) ? "true" : "false") << endl;
}