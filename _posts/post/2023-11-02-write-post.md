---
layout: post
description: >
  블로그를 사용하는 방법에 대한 전반적인 내용이다.
category: post
tags: post
author: hs_oh
comments: true
---
# 마크다운 [.md] 파일로 포스트 작성하기

### 글 쓰기

포스팅은 마크다운 문법을 따라 작성한다. 

포스팅 파일은 `.md` 확장자로 `_posts` 디렉토리 내에 생성한다. 

파일 이름은 `yyyy-mm-dd-[post name].md`로 작성한다. `[post name]` 부분은 자유롭게 써도 되나 날짜 부분은 꼭 지켜야한다. 

관리를 위해 `_posts` 안에 카테고리 디렉토리를 하나 더 만들어서 관리하도록 하자.

예를 들어 지금 작성하고 있는 파일을 post 카테고리에 넣고자 하여 `_post/post/2023-11-02-write-post.md` 로 만들었다. 

포스팅 양식은 다음과 같다. 

```markdown
---
layout: post
description: >
  블로그를 사용하는 방법에 대한 전반적인 내용이다.
category: post
tags: post
author: hs_oh
comments: true
---

# 마크다운 [.md] 파일로 포스트 작성하기

... 내용 ...
```

`---` 사이에는 머릿말을 쓰고, 그 아래에는 본문을 작성한다. 

머릿말의 각 부분에 대한 설명은 다음과 같다.

- `layout` : 레이아웃, 포스팅할 때는 `post` 를 쓰면 된다.
- `description` : 요약, 미리보기에 보여지는 글
- `category` : 카테고리
- `tags` : 태그
- `author` : 작성자, `_data/authors.yml`에 작성자 등록 후 사용한다.
- `comments` : 댓글 여부
- 포스팅 타이틀은 머릿글 아래에 `# [title]` 형식으로 작성한다. 

### _data/authors.yml
작성자 정보는 다음과 같이 작성하면 된다. 
```
# Another author (optional)
hs_oh:
  name: Hyung-Seok Oh
  email: hs_oh@korea.ac.kr
  about: |
    hs_oh

    expressive speech synthesis, emotional voice conversion
  picture: ./assets/img/profiles/hs_oh.jpeg
  # twitter: <username>
  github: https://github.com/hsoh0306
  accent_color: "#268bd2"
  accent_image:
    background: "#202020"
    overlay: false
```
`hs_oh`는 author의 아이디라고 생각하면 된다. 여기 쓰여진 값을 위에 작성자 정보에 쓰면 된다. 
프로필은 자유롭게 작성하면 된다.


### 문서 편집 도구

본문의 내용은 마크다운 문법에 따라서 자유롭게 작성하면 된다. 

다양한 문서 편집기를 사용할 수 있으니 편한거를 골라서 사용하면 되겠다.

간단하게 몇 가지 방법을 소개하고자 한다. 

1. **메모장**: 아무것도 설치하고 싶지 않을 경우에는 그냥 메모장을 사용해서 작성해도 된다. 
2. **VS Code**: 가볍고 다루기 편하다. 특히 마크다운 같은경우에는 미리 보기 기능을 지원해서 작성하는데 큰 도움이 된다. 

![Untitled](/assets/img/2023-11-02-write-post/fig1.png)

1. **Notion**: 노션이 익숙한 사람들은 노션으로 글을 작성해도 좋다. 확실히 다양한 기능이 있어서 글을 작성하는데 훨씬 수월하다. 지금 이 글 역시 노션으로 작성하고 있다. 

![Untitled](/assets/img/2023-11-02-write-post/fig2.png)

자유롭게 글을 작성한 후 `...` 누르고 `내보내기`를 `Markdown & CSV`으로 하면 `.zip` 파일로 다운받을 수 있다. 

![Untitled](/assets/img/2023-11-02-write-post/fig3.png)

`.zip`파일을 풀면 안에 `.md` 파일이랑 디렉토리가 있는 것을 확인 할 수 있다. 

![Untitled](/assets/img/2023-11-02-write-post/fig4.png)

`.md` 파일에는 노션으로 작성했던 본문 내용이 들어있고, 디렉토리에는 사용했던 이미지가 저장되어있다. 이 내용을 그대로 작성하면 된다. 

![Untitled](/assets/img/2023-11-02-write-post/fig5.png)

주의 해야할 점은 첨부 이미지들이 이런 식으로 되어있어서 그 부분만 수정하면 된다. 

### 이미지 첨부

마크다운에서 이미지를 첨부하는 방법은 다양하므로 편한대로 사용하면 된다. 여기서는 가장 간단한 형태의 방법만 소개하고자 한다. 

```markdown
![](/assets/img/[posting-name]/[image name].png)
```

우리는 `/assets/img` 디렉토리 아래에 포스팅에 필요한 이미지를 저장할 것이다. 관리를 위해 포스팅하고 있는 파일의 이름으로 디렉토리를 하나 만들고, 그 아래에 이미지를 저장하도록 하자. 

![Untitled](/assets/img/2023-11-02-write-post/fig6.png)

노션으로 글을 작성한 경우 이미지 부분을 이런 식으로 수정해서 사용하면 된다. 

### 카테고리 만들기

글을 작성할 때 원하는 카테고리가 없을 수도 있다. 그럴 경우에는 카테고리를 추가하면 된다. 현재 세팅에서 카테고리를 추가하려면 2개의 디렉토리를 확인하면 된다.

완전히 새로운 카테고리를 추가하고 싶을 때는 `_featured_categories` 아래에 파일을 추가하면 되고, 서브 메뉴를 추가하고 싶을 때는 `_featured_tags` 아래에 파일을 추가하면 된다. 

예를 들어 `_featured_categories` 아래에 `seminar` 카테고리를 추가하고자 하면 다음과 같이 작성하면 된다. 

- `_featured_categories/seminar.md`

```markdown
---
layout: list
type: category
title: Seminar
slug: seminar
sidebar: true
order: 3
description: >
  논문 읽고 정리해서 공유하기 
---
```

우리가 수정해야하는 부분은 `title`, `slug`, `order`, `description` 이다. 

`title`:  사이드메뉴에 직접적으로 보이는 부분이다.

`slug`: 카테고리의 고유 id, 보통 `title` 을 소문자화하고, 공백을 `-`으로 채워주는 것 같다. 

`order`: 사이드 메뉴의 순서

`description`: 소개글 

- `_featured_tags/text_to_speech.md`

```markdown
---
layout: tag-list
type: tag
slug: text-to-speech
category: seminar
sidebar: true
description: >
   Text-to-speech 관련 논문 
---

# title
```

서브 메뉴의 추가도 유사하다. 차이점은 `type`이 `tag`라는 점이랑 `category`를 지정해줘야한다는 것이다.

카테고리와 테그는 `featured_categories` `_featured_tags` 내에 `[category].md` 또는 `[tag].md` 파일을 확인하고, `slug`를 넣어야한다. 

### 포스팅 유의 사항

앞서 소개한 내용을 바탕으로 포스팅을 하는 방법은 다음과 같다. 

1. `_posts` 아래 원하는 카테고리 디렉토리에 파일을 작성한다. 카테고리가 없을 경우 생성한다. 파일 이름은 `yyyy-mm-dd-[file name].md`로 작성한다. 
2. 머릿글을 작성한다. 카테고리를 잘 확인하자. title은 머릿글 아래에 `# [title]`로 써준다. 
3. 원하는 문서 편집기를 이용하여 본문을 작성하도록 하자. 본문 내용은 마크다운 문법을 사용하고, 자유롭게 작성하면 된다. 
    1. 이미지를 첨부할 때는 `/assets/img/[file name]` 으로 디렉토리를 생성하고 그 안에 저장하도록 하자. 
4. 포스팅을 완료했으면 수정된 모든 파일을 `commit`하고 `push` 한다. 
    1. Github에 모든 업데이트가 반영되었는지 확인한다.
    2. 약 5분 정도 후에 블로그를 확인해 포스팅이 정상적으로 업로드 되었는지 확인한다. 
    3. 여러 사람이 동시에 같은 파일을 수정하게 될 경우 충돌이 나므로 주의하도록 하자.
