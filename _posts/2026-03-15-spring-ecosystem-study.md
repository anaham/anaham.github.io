---
layout: post
title: "Spring / Java 생태계 학습 정리"
date: 2026-03-15
categories: minoc
tags: [bss, spring, java, mybatis]
---

# Spring / Java 생태계 학습 정리

주말 동안 Spring Boot 3계층 아키텍처를 공부하며 큰 그림을 그려봤다.

BSS 레거시 전환 관점에서 MyBatis를 중심으로, Servlet부터 Spring Boot까지의 흐름을 정리.

---

## 인터랙티브 다이어그램

<iframe src="/assets/html/spring-ecosystem.html" style="width:100%; height:800px; border:none; border-radius:8px;" title="Spring Ecosystem"></iframe>

---

## 핵심 학습 내용

### 1. 계층의 진화

- **Servlet (1997~)**: HTTP 처리의 가장 기본 단위
- **Spring Framework (2003~)**: DispatcherServlet + IoC Container
- **Spring Boot (2014~)**: 자동 설정 + 내장 Tomcat

### 2. 3계층 아키텍처

- **Controller**: HTTP 요청 처리, URL 매핑만 담당
- **Service**: 비즈니스 로직, 규칙 판단
- **Repository**: DB 실제 액션 (MyBatis Mapper)

### 3. MyBatis vs JPA

**MyBatis (실습 완료):**
- SQL 직접 작성 → 복잡한 쿼리 제어 가능
- BSS 500개 Oracle 프로시저 같은 레거시 전환에 적합

**JPA (개념 수준):**
- SQL 자동 생성 → 단순 CRUD 신규 개발에 적합
- 복잡한 레거시 전환엔 한계

### 4. 핵심 개념

**IoC / DI:**
- Spring이 객체 생성·주입 담당
- 개발자는 `@Service`, `@Mapper` 선언만

**느슨한 결합 (Loose Coupling):**
- 인터페이스로 계층 연결
- DB 교체해도 위 계층 불변

**Spring Bean:**
- Spring이 관리하는 객체 단위
- ApplicationContext가 Bean들을 담고 연결

---

## 학습 환경

- **Spring Boot**: 3.2.5
- **빌드 도구**: Gradle
- **DB 연동**: MyBatis + PostgreSQL
- **실습 기간**: Week 6~8 (우리학습프로그램)

---

## 다음 단계

이제 큰 그림이 잡혔으니:
1. BSS 미니어처(Minoc)에 Spring Boot 적용
2. 3계층 아키텍처로 리팩토링
3. MyBatis Mapper로 DB 연동 실전 구현

---

*Spring / Java 생태계 — 주말 학습 완료 2026-03-15*
