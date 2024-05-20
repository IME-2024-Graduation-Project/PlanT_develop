import React, { useState } from 'react';
import { Link } from 'react-router-dom';

function Test() {
  // 사용자가 입력한 친환경 레벨과 도보 시간을 관리할 상태
  const [ecoLevel, setEcoLevel] = useState('');
  const [walkingTime, setWalkingTime] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleEcoLevelChange = (event) => {
    setEcoLevel(event.target.value);
  };

  const handleWalkingTimeChange = (event) => {
    setWalkingTime(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setSubmitted(true);
  };

  const ecoLevelDescriptions = {
    1: "최소한의 노력으로도 친환경적인 선택을 할 수 있는 단계입니다.",
    2: "가능한 경우 친환경적인 선택을 하려고 노력하는 단계입니다.",
    3: "친환경적인 선택을 지속적으로 실천하고 있는 단계입니다.",
    4: "일상생활에서 친환경을 고려하며 적극적으로 실천하는 단계입니다.",
    5: "모든 가능한 방법으로 친환경을 실천하며 환경에 대한 경각심을 가지고 있는 단계입니다."
  };

  return (
    <div>
      <h1>친환경 레벨과 도보 시간 입력</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="ecoLevel">친환경 레벨 선택:</label>
          <select id="ecoLevel" value={ecoLevel} onChange={handleEcoLevelChange}>
            <option value="">친환경 레벨을 선택하세요</option>
            <option value="1">레벨 1</option>
            <option value="2">레벨 2</option>
            <option value="3">레벨 3</option>
            <option value="4">레벨 4</option>
            <option value="5">레벨 5</option>
          </select>
          {submitted && !ecoLevel && <p>친환경 레벨을 선택해주세요.</p>}
          {ecoLevel && <p>{ecoLevelDescriptions[ecoLevel]}</p>}
        </div>
        <div>
          <label htmlFor="walkingTime">도보 시간 입력 (분):</label>
          <input
            type="number"
            id="walkingTime"
            value={walkingTime}
            onChange={handleWalkingTimeChange}
          />
          {submitted && !walkingTime && <p>도보 시간을 입력해주세요.</p>}
        </div>
        <button type="submit">확인</button>
      </form>
      {submitted && ecoLevel && walkingTime && (
        <p>친환경 레벨: {ecoLevel}, 도보 시간: {walkingTime}분</p>
      )}
      <Link to="/place">다음</Link>
    </div>
  );
}

export default Test;
