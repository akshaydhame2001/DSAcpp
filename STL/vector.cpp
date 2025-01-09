#include <bits/stdc++.h>
using namespace std;

void explainVector()
{
    // Creating a vector of integers
    vector<int> v;

    // Adding elements to the vector
    v.push_back(1);    // Adds 1 to the vector
    v.emplace_back(2); // Adds 2 to the vector (similar to push_back but more efficient)

    // Creating a vector of pairs
    vector<pair<int, int>> vec;
    vec.push_back({1, 2});  // Adds pair (1, 2) using push_back
    vec.emplace_back(3, 4); // Adds pair (3, 4) using emplace_back (slightly more efficient)

    // Creating a vector with all elements initialized to a specific value
    vector<int> v1(5, 100); // A vector of size 5 with all elements initialized to 100

    // Creating an empty vector with a specific size (default values are 0 for integers)
    vector<int> v2(5); // A vector of size 5 with default value 0

    // Copying a vector
    vector<int> v3(v1); // Copies all elements from v1 to v3

    // Printing vectors for demonstration
    cout << "Vector v: ";
    for (int i : v)
        cout << i << " ";
    cout << "\n";

    cout << "Vector of pairs vec: ";
    for (auto p : vec)
        cout << "(" << p.first << ", " << p.second << ") ";
    cout << "\n";

    cout << "Vector v1: ";
    for (int i : v1)
        cout << i << " ";
    cout << "\n";

    cout << "Vector v2: ";
    for (int i : v2)
        cout << i << " ";
    cout << "\n";

    cout << "Vector v3 (copied from v1): ";
    for (int i : v3)
        cout << i << " ";
    cout << "\n";
}

int main()
{
    explainVector();
    return 0;
}
