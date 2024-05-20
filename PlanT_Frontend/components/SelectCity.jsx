import React, { useState } from 'react';
import { Link } from 'react-router-dom';

function SelectCity() {
  const cities = [
    { code: '1', city: '서울' },
    { code: '2', city: '도쿄' },
    { code: '3', city: '뉴욕' },
    { code: '2', city: '파리' },
    { code: '3', city: '런던' },
  ];

  const [query, setQuery] = useState("");
  const [searchedCities, setSearchedCities] = useState([]);
  const [selectedCities, setSelectedCities] = useState([]);

  const handleSearch = () => {
    const filteredselectedCity = cities.filter(({ city }) =>
    city.includes(query)
    );
    setSearchedCities(filteredselectedCity);
  };

  const handleSelect = (cityName) => {
    if (!selectedCities.some(cities => cities.city === cityName)) {
      setSelectedCities(prevSelectedCities => [
        ...prevSelectedCities,
        cities.find(cities => cities.city === cityName)
      ]);
    }
  };

  return (
    <main>
      <h1>Plan🌱</h1>
      <div className='Travel_City'>
        <p>여행 지역 검색</p>
        <input
          type="search"
          placeholder="여행할 지역 검색"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button onClick={handleSearch}>검색</button>
      </div>
      <div className='Searched_selectedCity'>
        <header>총 {searchedCities.length}개의 도시가 검색되었습니다.</header>
        <ul>
          {searchedCities.map(cities => (
            <li key={cities.code}>
              {cities.city}
              <button onClick={() => handleSelect(cities.city)}>선택</button>
            </li>
          ))}
        </ul>
        <h2>선택된 지역 목록</h2>
        <ul>
          {selectedCities.map((cities, index) => (
            <li key={index}>{cities.city}</li>
          ))}
        </ul>
      </div>
      {selectedCities.length >= 2 ? (
        // <Link to={`/movedate/${selectedCities.length}`}>다음</Link>
        <Link to="/movedate">다음</Link>
      ) : (
        <Link to="/ecolevel">다음</Link>
      )}
    </main>
  );
}

export default SelectCity;