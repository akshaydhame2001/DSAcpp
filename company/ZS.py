def magic_value(arr):
    """
    Calculate the difference between good integers (numbers whose index doesn't change after sorting)
    and bad integers (remaining numbers) in an array.
    
    Args:
        arr (List[int]): Input array
    Returns:
        int: Difference between sum of good and bad integers
    """
    # Create sorted array
    sorted_arr = sorted(arr)
    
    good_sum = 0
    bad_sum = 0
    
    # Compare original and sorted arrays
    for i in range(len(arr)):
        if arr[i] == sorted_arr[i]:
            good_sum += arr[i]  # Number is at same position, so it's good
        else:
            bad_sum += arr[i]   # Number changed position, so it's bad
            
    return abs(good_sum - bad_sum)

def longest_non_decreasing_subarray(arr):
    """
    Find the length of longest non-decreasing subarray in given array.
    A non-decreasing array is where arr[i] <= arr[i+1] for all valid i.
    
    Args:
        arr (List[int]): Input array
    Returns:
        int: Length of longest non-decreasing subarray
    """
    if not arr:
        return 0
        
    max_length = 1  # Minimum length is 1
    current_length = 1
    
    # Traverse array looking for non-decreasing sequences
    for i in range(1, len(arr)):
        if arr[i] >= arr[i-1]:  # Current element is greater or equal to previous
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1   # Reset length when sequence breaks
            
    return max_length

def main():
    # Get number of elements
    n = int(input())
    # Get array elements
    arr = list(map(int, input().split()))
    
    # Ensure array length matches n
    if len(arr) != n:
        print("Error: Number of elements doesn't match input size")
        return
    
    # Calculate and print results
    print(magic_value(arr))
    print(longest_non_decreasing_subarray(arr))

if __name__ == "__main__":
    main()