import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
    return (
      <main>
        <h1>Plan🌱</h1>
        <p>탄소 배출량을 저감하는 친환경 여행 코스를 추천해드려요!</p>
        <p>로그인 페이지</p>
        <Link to="/date">시작하기</Link>
      </main>
    );
  }

export default Home;