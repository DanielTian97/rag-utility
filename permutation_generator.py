import itertools

# def get_combinations(lst): # creating a user-defined method
#    combination = [] # empty list 
#    for r in range(1, len(lst) + 1):
#       # to generate combination
#       combination.extend(itertools.combinations(lst, r))
#    return combination

def construct_p_name(l):
    name = ''
    for i in l:
        name += f'_{i}'
    return name

def get_permutation(lst, p_len): # creating a user-defined method
    index = range(len(lst))
    if(p_len > len(lst)):
        print("Permutation length should be less than or equal to the length of the list.")
        return []
    permutations = list(itertools.permutations(index, p_len))
    
    permutation_book = {}
    for p in permutations:
        p_name = construct_p_name(p)
        permutation_book.update({p_name:[lst[i] for i in list(p)]})
    return permutation_book

def get_permutation_simple(lst): # creating a user-defined method
    permutations = list(itertools.permutations(lst))
    return permutations

if __name__=="__main__":
    print("PERMUTATION TEST:")
    l1 = [3, 2]
    print(f'List = {l1}')
    all_combinations_1 = get_permutation(l1, len(l1)) # method call
    print(all_combinations_1)