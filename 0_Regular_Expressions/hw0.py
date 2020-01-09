import re

EMAIL_RE = re.compile(r"[^a-zA-z0-9\.\!\#\$\%\&\'\*\+\-\/\=\?\^\_\`\{\|\}\~]"
    r"([a-zA-z0-9\!\#\$\%\&\'\*\+\-\/\=\?\^\_\`\{\|\}\~]+\.?[a-zA-z0-9\!\#\$\%\&\'\*\+\-\/\=\?\^\_\`\{\|\}\~]+"
    r"|[a-zA-z0-9\!\#\$\%\&\'\*\+\-\/\=\?\^\_\`\{\|\}\~]+)( \[at\] | /at/ | @ |@@|@| at )([a-zA-z0-9\-]+)"
    r"( \[dot\] | /dot/ | dot | \. |\.)(gov|com|net|org)[^a-zA-z0-9\.\!\#\$\%\&\'\*\+\-\/\=\?\^\_\`\{\|\}\~]")

with open("W20_webpages.txt","r") as f:
    i = 0
    with open("email-outputs.csv","w") as g:
        g.write("Id,Category\n")
        for line in f:
            # if i % 1000 == 0:
            #     print(i)
            res = EMAIL_RE.findall(line)
            g.write(str(i)+","+ (res[0][0]+"@"+res[0][2]+"."+res[0][-1] if res else "None") +"\n")
            i += 1
