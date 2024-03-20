def save_list_to_file(path, thelist):
    """
    Writes each item of the list into the specified file
    """
    with open(path, 'a') as f:
        for item in thelist:
            f.write(f"{item},")
        f.write("\n")