import React, { useState } from 'react';
import { Link } from 'react-router-dom';


function DateRangePicker() {
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleStartDateChange = (event) => {
    setStartDate(event.target.value);
  };

  const handleEndDateChange = (event) => {
    setEndDate(event.target.value);
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    if (startDate && endDate) {
      setSubmitted(true);
    } else {
      alert('여행 시작일과 종료일을 입력하세요.');
    }
  };


  return (
    <div>
      <h1>여행 시작일과 종료일 입력</h1>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="startDate">여행 시작일:</label>
          <input
            type="date"
            id="startDate"
            value={startDate}
            onChange={handleStartDateChange}
          />
        </div>
        <div>
          <label htmlFor="endDate">여행 종료일:</label>
          <input
            type="date"
            id="endDate"
            value={endDate}
            onChange={handleEndDateChange}
          />
        </div>
        <button type="submit">확인</button>
      </form>
      {submitted && startDate && endDate && (
        <p>여행 시작일: {startDate}, 여행 종료일: {endDate}</p>
      )}
      <Link to="/city">다음</Link>
    </div>
  );
}

export default DateRangePicker;