#include <bits/stdc++.h>
using namespace std;

// character hashing

int main()
{
    string s;
    cout << "Enter string: ";
    cin >> s;

    // precompute
    int hash[26] = {0};
    for (int i = 0; i < s.size(); i++)
    {
        hash[s[i] - 'a']++;
    }

    int q;
    cout << "Number of queries: ";
    cin >> q;
    cout << "Queries: ";
    while (q--)
    {
        char c;
        cin >> c;
        cout << hash[c - 'a'] << endl;
    }
    return 0;
}

// for extend full ascii chars, hash[256] and remove -'a'