#include <bits/stdc++.h>
using namespace std;

void explainList()
{
    list<int> ls;
    ls.push_back(2);     // Add 2 at the back
    ls.emplace_back(4);  // Add 4 at the back (more efficient than push_back)
    ls.push_front(5);    // Add 5 at the front
    ls.emplace_front(6); // Add 6 at the front (more efficient than push_front)

    // Output the list
    for (int x : ls)
    {
        cout << x << " "; // Print each element separated by a space
    }
    cout << endl;
    // rest functions same as vector
    // begin, end, rbegin, rend, clear, insert, size, swap
}

int main()
{
    explainList();
    return 0;
}
