#include <bits/stdc++.h>
using namespace std;

void explainVector()
{
    cout << "=== Demonstrating Vector Operations ===\n";

    // Creating a vector of integers
    vector<int> v;

    // Adding elements to the vector
    v.push_back(1);    // Adds 1 to the vector
    v.emplace_back(2); // Adds 2 to the vector (similar to push_back but more efficient)
    cout << "Initial vector (after push_back and emplace_back): ";
    for (int i : v)
        cout << i << " ";
    cout << "\n";

    // Creating a vector of pairs
    vector<pair<int, int>> vec;
    vec.push_back({1, 2});  // Adds pair (1, 2) using push_back
    vec.emplace_back(3, 4); // Adds pair (3, 4) using emplace_back
    cout << "Vector of pairs: ";
    for (auto p : vec)
        cout << "(" << p.first << ", " << p.second << ") ";
    cout << "\n";

    // Creating a vector with specific size and initial value
    vector<int> v1(5, 100); // A vector of size 5 with all elements initialized to 100
    cout << "Vector v1 (size 5, initialized to 100): ";
    for (int i : v1)
        cout << i << " ";
    cout << "\n";

    // Creating an empty vector with default values
    vector<int> v2(5); // A vector of size 5 with default value 0
    cout << "Vector v2 (size 5, default values): ";
    for (int i : v2)
        cout << i << " ";
    cout << "\n";

    // Copying a vector
    vector<int> v3(v1); // Copies all elements from v1 to v3
    cout << "Vector v3 (copied from v1): ";
    for (int i : v3)
        cout << i << " ";
    cout << "\n";

    // Accessing elements
    if (!v.empty())
    {
        cout << "Accessing elements of vector v: ";
        cout << "First element: " << v[0] << ", ";
        cout << "Using at(): " << v.at(0) << ", ";
        cout << "Last element: " << v.back() << "\n";
    }

    // Using a loop with iterators
    cout << "Elements of vector v using iterators: ";
    for (auto it = v.begin(); it != v.end(); ++it)
        cout << *it << " ";
    cout << "\n";

    // Erasing elements
    v = {10, 20, 12, 23, 35}; // Reassign values for demonstration
    v.erase(v.begin() + 1);   // Removes the second element
    cout << "Vector after erasing second element: ";
    for (int i : v)
        cout << i << " ";
    cout << "\n";

    v.erase(v.begin() + 1, v.begin() + 3); // Removes range [2nd, 4th)
    cout << "Vector after erasing a range: ";
    for (int i : v)
        cout << i << " ";
    cout << "\n";

    // Inserting elements
    v = {100};                      // Reset vector
    v.insert(v.begin(), 300);       // Insert 300 at the beginning
    v.insert(v.begin() + 1, 2, 10); // Insert two 10s at position 1
    cout << "Vector after insertions: ";
    for (int i : v)
        cout << i << " ";
    cout << "\n";

    vector<int> copy(2, 5);                        // A new vector {5, 5}
    v.insert(v.begin(), copy.begin(), copy.end()); // Insert another vector at the beginning
    cout << "Vector after inserting another vector: ";
    for (int i : v)
        cout << i << " ";
    cout << "\n";

    // Swapping vectors
    v1 = {10, 20};
    v2 = {30, 40};
    cout << "Before swap:\n";
    cout << "v1: ";
    for (int i : v1)
        cout << i << " ";
    cout << "\nv2: ";
    for (int i : v2)
        cout << i << " ";
    cout << "\n";

    v1.swap(v2);
    cout << "After swap:\n";
    cout << "v1: ";
    for (int i : v1)
        cout << i << " ";
    cout << "\nv2: ";
    for (int i : v2)
        cout << i << " ";
    cout << "\n";

    // Clearing the vector
    v.clear();
    cout << "Is vector v empty after clearing? " << (v.empty() ? "Yes" : "No") << "\n";
}

int main()
{
    explainVector();
    return 0;
}

/*

Initial vector (after push_back and emplace_back): 1 2
Vector of pairs: (1, 2) (3, 4)
Vector v1 (size 5, initialized to 100): 100 100 100 100 100
Vector v2 (size 5, default values): 0 0 0 0 0
Vector v3 (copied from v1): 100 100 100 100 100
Accessing elements of vector v: First element: 1, Using at(): 1, Last element: 2
Elements of vector v using iterators: 1 2
Vector after erasing second element: 10 12 23 35
Vector after erasing a range: 10 35
Vector after insertions: 300 10 10 100
Vector after inserting another vector: 5 5 300 10 10 100
Before swap:
v1: 10 20
v2: 30 40
After swap:
v1: 30 40
v2: 10 20
Is vector v empty after clearing? Yes

*/