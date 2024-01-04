---
layout: post
category: post
tags: post
author: hs_oh
comments: true
---

# 파이썬으로 네이버웍스 메일 보내기

```
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def sendMail(from_, pw_, to_, title_, content_):
    message = MIMEMultipart()
    message["Subject"] = title_
    message["From"] = from_
    message["To"] = to_

    mimetext = MIMEText(content_, "html")
    message.attach(mimetext)

    server = smtplib.SMTP("smtp.worksmobile.com", 587)  # 네이버웍스 메일 서버, SMTP 포트
    server.ehlo()
    server.starttls()
    server.login(from_, pw_)
    server.sendmail(message["From"], to_, message.as_string())
    server.quit()


if __name__ == "__main__":
    from_ = ""  # 보내는 사람 메일
    pw_ = ""  # 외부 접근용 비밀번호
    to_ = ""  # 받는 사람 메일
    name_ = ""  # 이름
    title_ = ""  # 메일 제목
    content_ = ""  # 메일 본문

    sendMail(from_, pw_, to_, title_, content_)

```

### 네이버 웍스 외부 접근 
외부에서 이메일을 보내려면 외부 접근용 비밀번호가 필요하다. 학교 계정은 네이버웍스로 연동되어 있기 때문에 네이버 웍스에서 외부 비밀번호를 생성하면 된다. 

1. 환경 설정 -> 메일 -> 고급 설정 -> IMAP/SMTP -> 외부 앱 비밀번호 생성하기 -> 새 비밀번호 생성 -> 복사

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/cec2bd78-ed28-4665-a99a-462d5ffd626c)

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/eb47ea57-4294-46f3-b77a-21c3b86330f8)

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/31188a0e-e25b-432b-ba66-a0e385309856)

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/17d17cc0-dc9b-4568-ab31-2f657aad192d)

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/b3e2325c-1314-4d04-a45f-35c1e28f81b1)

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/43509be1-4b99-417b-9264-ecbda88b59a7)

![image](https://github.com/speech-team-korea/speech-team-korea.github.io/assets/43984708/05f3b51e-7f60-4862-8f62-c9ed392e76bf)

