import sys

def parse_line(line):
    hash, repo_link = line.split("\t")
    parts = repo_link.split("/")
    user = parts[3]
    repo = parts[4]
    return user, repo, hash

def main():
    # download_link = "https://codeload.github.com/{user}/{repo}/zip/{hash}"
    download_link = "https://codeload.github.com/{}/{}/zip/{}"

    for line in sys.stdin:
        line = line.strip()
        if line == "":
            continue

        print(download_link.format(*parse_line(line)))

if __name__ == "__main__":
    main()