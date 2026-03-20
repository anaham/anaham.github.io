# SEO 설정 가이드

## 1. _config.yml 업데이트

**현재 위치:** `_config.yml.seo` (예시 파일)

**수정 방법:**
```bash
# 백업
cp _config.yml _config.yml.backup

# 예시 파일로 교체
cp _config.yml.seo _config.yml

# Git 커밋
git add _config.yml robots.txt
git commit -m "Add: SEO 설정 (sitemap, seo-tag, robots.txt)"
git push
```

**추가된 플러그인:**
- `jekyll-sitemap`: sitemap.xml 자동 생성
- `jekyll-seo-tag`: 메타태그 자동 생성 (Open Graph, Twitter Card 등)

**추가된 설정:**
- `lang: ko_KR`: 한국어 명시
- `timezone: Asia/Seoul`: 타임존 설정

---

## 2. robots.txt 생성 완료 ✅

**파일:** `robots.txt` (루트 디렉토리)

**내용:**
```
User-agent: *
Allow: /

Sitemap: https://anaham.github.io/sitemap.xml
```

---

## 3. Google Search Console 등록

### 3-1. 사이트 등록
1. https://search.google.com/search-console 접속
2. "속성 추가" 클릭
3. **URL 접두어** 선택: `https://anaham.github.io`
4. 소유권 확인 방법 선택

### 3-2. 소유권 확인 (GitHub Pages 권장)

**방법 A: HTML 파일 업로드**
1. Google이 제공하는 `google*.html` 파일 다운로드
2. GitHub 레포 루트에 업로드
3. Push → 확인 클릭

**방법 B: HTML 태그 (추천)**
1. Google이 제공하는 `<meta>` 태그 복사
2. `_includes/head.html` 파일에 추가:
   ```html
   <meta name="google-site-verification" content="YOUR_CODE_HERE" />
   ```
3. Push → 확인 클릭

### 3-3. Sitemap 제출
1. Search Console → Sitemap 메뉴
2. `sitemap.xml` 입력
3. "제출" 클릭
4. 상태가 "성공"으로 바뀌면 완료!

---

## 4. Naver 웹마스터 도구 (선택)

한국 검색은 Naver도 중요:

1. https://searchadvisor.naver.com/ 접속
2. 사이트 등록: `https://anaham.github.io`
3. HTML 태그 또는 파일 업로드로 소유권 확인
4. Sitemap 제출: `https://anaham.github.io/sitemap.xml`

---

## 5. 확인 사항

### sitemap.xml 생성 확인
```
https://anaham.github.io/sitemap.xml
```

- 빌드 후 확인 (첫 배포 후 자동 생성됨)
- 모든 포스트 URL이 나열되어야 함

### robots.txt 확인
```
https://anaham.github.io/robots.txt
```

- User-agent: *
- Sitemap 경로 포함

---

## 6. 추가 최적화 (선택)

### 6-1. 메타 디스크립션
각 포스트 front matter에 추가:
```yaml
---
title: "제목"
description: "이 글은 Spring/Java 생태계를 다룹니다..."
---
```

### 6-2. Open Graph 이미지
대표 이미지 설정:
```yaml
---
title: "제목"
image: /assets/images/og-image.png
---
```

### 6-3. 구조화된 데이터
`jekyll-seo-tag`가 자동 생성:
- Article schema
- BreadcrumbList
- Organization

---

## 7. 효과 측정

### Google Search Console
- **실적 탭**: 클릭수, 노출수, CTR, 평균 게재순위
- **검사 도구**: URL 색인 상태 확인
- **커버리지**: 색인된 페이지 수

### 예상 소요 시간
- **Google**: 1~2주 (빠르면 며칠)
- **Naver**: 며칠~1주

---

## 8. 빠른 체크리스트

- [ ] `_config.yml` 업데이트 (sitemap, seo-tag 플러그인)
- [ ] `robots.txt` 생성
- [ ] Git push
- [ ] GitHub Pages 빌드 확인 (Actions 탭)
- [ ] `sitemap.xml` 생성 확인
- [ ] Google Search Console 등록
- [ ] 소유권 확인
- [ ] Sitemap 제출
- [ ] (선택) Naver 웹마스터 도구 등록

---

**작성일:** 2026-03-15  
**작성자:** Wisp ✨
