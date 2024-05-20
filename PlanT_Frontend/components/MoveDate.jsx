import React, { useState } from 'react';
import { Link } from 'react-router-dom';

function MoveDate() {
  const [moveDates, setMoveDates] = useState('');
  const [transportation, setTransportation] = useState('');
  const [submitted, setSubmitted] = useState(false);
  const [departureLocation, setDepartureLocation] = useState('');
  const [arrivalLocation, setArrivalLocation] = useState('');

  const handleMoveDateChange = (event) => {
    setMoveDates(event.target.value);
  };

  const handleDepartureLocationChange = (event) => {
    setDepartureLocation(event.target.value);
  };

  const handleArrivalLocationChange = (event) => {
    setArrivalLocation(event.target.value);
  };

  const handleTransportationChange = (event) => {
    setTransportation(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (moveDates && departureLocation && arrivalLocation && transportation) {
      setSubmitted(true);
    } else {
      alert('여행 이동 날짜, 출발 지역, 도착 지역, 이동 수단을 모두 입력하세요.');
    }
  };

  // 임시로 사용할 이동 수단 옵션 리스트
  const transportationOptions = [
    '서울역',
    '용산역',
    '대전역',
    '동서울터미널',
  ];

  return (
    <div>
      <h1>여행 이동 날짜 입력</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="moveDates">이동 날짜:</label>
          <input
            type="date"
            id="moveDates"
            value={moveDates}
            onChange={handleMoveDateChange}
          />
        </div>
        <div>
          <label htmlFor="departureLocation">출발 지역:</label>
          <input
            type="text"
            id="departureLocation"
            value={departureLocation}
            onChange={handleDepartureLocationChange}
          />
        </div>
        <div>
          <label htmlFor="arrivalLocation">도착 지역:</label>
          <input
            type="text"
            id="arrivalLocation"
            value={arrivalLocation}
            onChange={handleArrivalLocationChange}
          />
        </div>
        <div>
          <label htmlFor="transportation">이동 수단:</label>
          <select id="transportation" onChange={handleTransportationChange} value={transportation}>
            <option value="">이동 수단을 선택하세요</option>
            {transportationOptions.map((option, index) => (
              <option key={index} value={option}>{option}</option>
            ))}
          </select>
        </div>
        <button type="submit">확인</button>
      </form>
      {submitted && moveDates && departureLocation && arrivalLocation && transportation && (
        <div>
          <p>이동 날짜: {moveDates}</p>
          <p>출발 지역: {departureLocation}</p>
          <p>도착 지역: {arrivalLocation}</p>
          <p>이동 수단: {transportation}</p>
        </div>
      )}
      <Link to="/ecolevel">다음</Link>
    </div>
  );
}

export default MoveDate;
