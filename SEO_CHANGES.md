# SEO 설정 변경사항 요약

## 생성/수정된 파일

### 1. robots.txt ✅ (새로 생성)
- Google/Naver 크롤러 허용
- sitemap.xml 경로 명시

### 2. _config.yml.seo (예시 파일)
- **중요:** 이 파일을 `_config.yml`로 복사해서 기존 파일 교체 필요
- 추가된 플러그인:
  - `jekyll-sitemap`: sitemap.xml 자동 생성
  - `jekyll-seo-tag`: 메타태그 자동 생성
- 추가된 설정:
  - `lang: ko_KR`
  - `timezone: Asia/Seoul`

### 3. _includes/head.html ✅ (새로 생성)
- SEO 태그 자동 삽입 (`{% seo %}`)
- Google/Naver 소유권 확인 태그 자리 준비

### 4. _layouts/default.html ✅ (새로 생성)
- head.html include
- 기본 레이아웃 (remote theme 오버라이드)

### 5. SEO_GUIDE.md ✅
- 전체 가이드 문서
- Google Search Console 등록 방법
- Naver 웹마스터 도구 등록 방법

---

## 적용 방법

### Step 1: _config.yml 업데이트
```bash
cd /home/dupre/.openclaw/workspace/anaham.github.io

# 백업
cp _config.yml _config.yml.backup

# 새 설정 적용
cp _config.yml.seo _config.yml
```

### Step 2: Git 커밋 & 푸시
```bash
git add .
git commit -m "Add: SEO 최적화 (sitemap, seo-tag, robots.txt, meta tags)"
git push
```

### Step 3: 빌드 확인
- GitHub Actions에서 초록불 확인
- https://anaham.github.io/sitemap.xml 접속 확인
- https://anaham.github.io/robots.txt 접속 확인

### Step 4: Google Search Console 등록
- SEO_GUIDE.md의 "3. Google Search Console 등록" 참조
- sitemap.xml 제출

---

## 예상 효과

### 즉시
- ✅ sitemap.xml 자동 생성
- ✅ robots.txt 검색엔진 가이드
- ✅ Open Graph, Twitter Card 자동 생성
- ✅ 구조화된 데이터 (Schema.org)

### 1~2주 후
- 🔍 Google 검색 노출 시작
- 📊 Search Console에서 통계 확인

### 추가 작업 (선택)
- 각 포스트에 `description:` 추가
- 대표 이미지 `image:` 설정
- Naver 웹마스터 도구 등록

---

**작성일:** 2026-03-15  
**준비 완료!** 이제 Shamino가 직접 적용하면 됨.
