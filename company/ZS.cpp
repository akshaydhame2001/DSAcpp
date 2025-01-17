/* Calculate the "magic value" differnece of good integers(sum of integers whose index not changed after sorting)
and bad integers(sum of remaining integers)
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// Function to calculate the "magic value"
int magic_value(const vector<int> &arr)
{
    // Create a sorted copy of the array
    vector<int> sorted_arr = arr;
    sort(sorted_arr.begin(), sorted_arr.end());

    int good_sum = 0;
    int bad_sum = 0;

    // Compare the original and sorted arrays
    for (size_t i = 0; i < arr.size(); i++)
    {
        if (arr[i] == sorted_arr[i])
        {
            good_sum += arr[i]; // Good integer
        }
        else
        {
            bad_sum += arr[i]; // Bad integer
        }
    }

    return abs(good_sum - bad_sum);
}

// Function to find the length of the longest non-decreasing subarray
int longest_non_decreasing_subarray(const vector<int> &arr)
{
    if (arr.empty())
    {
        return 0;
    }

    int max_length = 1; // Minimum length of a non-decreasing subarray
    int current_length = 1;

    // Traverse the array to find the longest non-decreasing sequence
    for (size_t i = 1; i < arr.size(); i++)
    {
        if (arr[i] >= arr[i - 1])
        {
            current_length++;
            max_length = max(max_length, current_length);
        }
        else
        {
            current_length = 1; // Reset the length
        }
    }

    return max_length;
}

int main()
{
    int n;

    // Get the number of elements
    cin >> n;

    // Get the array elements
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }

    // Ensure the array length matches `n`
    if (arr.size() != static_cast<size_t>(n))
    {
        cout << "Error: Number of elements doesn't match input size" << endl;
        return 1;
    }

    // Calculate and print results
    cout << magic_value(arr) << endl;
    cout << longest_non_decreasing_subarray(arr) << endl;

    return 0;
}
