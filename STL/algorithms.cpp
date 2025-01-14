#include <algorithm>
#include <iostream>
#include <vector>
#include <numeric> // For accumulate operation
using namespace std;

int main()
{
    // Initializing vector with array values
    int arr[] = {10, 20, 5, 23, 42, 15, 20};
    int n = sizeof(arr) / sizeof(arr[0]);
    vector<int> vect(arr, arr + n);

    // Display original vector
    cout << "Original Vector: ";
    for (int i = 0; i < n; i++)
    {
        cout << vect[i] << " ";
    }
    cout << endl;

    // Non-Manipulating Algorithms:

    // Sorting in ascending order
    sort(vect.begin(), vect.end());
    cout << "\nSorted in Ascending Order: ";
    for (int i = 0; i < n; i++)
    {
        cout << vect[i] << " ";
    }

    // Sorting in descending order
    sort(vect.begin(), vect.end(), greater<int>());
    cout << "\nSorted in Descending Order: ";
    for (int i = 0; i < n; i++)
    {
        cout << vect[i] << " ";
    }

    // Reversing the vector
    reverse(vect.begin(), vect.end());
    cout << "\nReversed Vector: ";
    for (int i = 0; i < n; i++)
    {
        cout << vect[i] << " ";
    }

    // Maximum element
    cout << "\nMaximum Element: " << *max_element(vect.begin(), vect.end());

    // Minimum element
    cout << "\nMinimum Element: " << *min_element(vect.begin(), vect.end());

    // Accumulate the sum of elements
    cout << "\nSum of Elements: " << accumulate(vect.begin(), vect.end(), 0) << endl;

    // Counting occurrences of 20
    cout << "\nOccurrences of 20: " << count(vect.begin(), vect.end(), 20) << endl;

    // Finding element 5
    if (find(vect.begin(), vect.end(), 5) != vect.end())
    {
        cout << "\nElement 5 Found in Vector";
    }
    else
    {
        cout << "\nElement 5 Not Found in Vector";
    }

    // Binary search for 23 (works only on sorted vector)
    sort(vect.begin(), vect.end()); // Make sure vector is sorted
    if (binary_search(vect.begin(), vect.end(), 23))
    {
        cout << "\nElement 23 Found using Binary Search";
    }
    else
    {
        cout << "\nElement 23 Not Found using Binary Search";
    }

    // Lower bound for element 20 (first element not less than 20)
    auto lower = lower_bound(vect.begin(), vect.end(), 20);
    cout << "\nLower Bound for 20 is at index: " << distance(vect.begin(), lower);

    // Upper bound for element 20 (first element greater than 20)
    auto upper = upper_bound(vect.begin(), vect.end(), 20);
    cout << "\nUpper Bound for 20 is at index: " << distance(vect.begin(), upper);

    // Distance between first element and maximum element
    cout << "\nDistance from first to maximum element: ";
    cout << distance(vect.begin(), max_element(vect.begin(), vect.end())) << endl;

    // Manipulating Algorithms:

    // Erase element '10' from vector
    vect.erase(find(vect.begin(), vect.end(), 10));
    cout << "\nVector after erasing element 10: ";
    for (int i = 0; i < vect.size(); i++)
    {
        cout << vect[i] << " ";
    }

    // Remove duplicates (requires sorted vector)
    vect.erase(unique(vect.begin(), vect.end()), vect.end());
    cout << "\nVector after removing duplicates: ";
    for (int i = 0; i < vect.size(); i++)
    {
        cout << vect[i] << " ";
    }

    // Next permutation
    next_permutation(vect.begin(), vect.end());
    cout << "\nVector after next permutation: ";
    for (int i = 0; i < vect.size(); i++)
    {
        cout << vect[i] << " ";
    }

    // Previous permutation
    prev_permutation(vect.begin(), vect.end());
    cout << "\nVector after previous permutation: ";
    for (int i = 0; i < vect.size(); i++)
    {
        cout << vect[i] << " ";
    }

    // Erase the first element
    vect.erase(vect.begin());
    cout << "\nVector after erasing the first element: ";
    for (int i = 0; i < vect.size(); i++)
    {
        cout << vect[i] << " ";
    }

    return 0;
}

/*
Original Vector: 10 20 5 23 42 15 20

Sorted in Ascending Order: 5 10 15 20 20 23 42
Sorted in Descending Order: 42 23 20 20 15 10 5
Reversed Vector: 5 10 15 20 20 23 42
Maximum Element: 42
Minimum Element: 5
Sum of Elements: 135

Occurrences of 20: 2

Element 5 Found in Vector
Element 23 Found using Binary Search
Lower Bound for 20 is at index: 3
Upper Bound for 20 is at index: 5
Distance from first to maximum element: 6

Vector after erasing element 10: 5 15 20 20 23 42
Vector after removing duplicates: 5 15 20 23 42
Vector after next permutation: 5 15 20 42 23
Vector after previous permutation: 5 15 20 23 42
Vector after erasing the first element: 15 20 23 42
*/