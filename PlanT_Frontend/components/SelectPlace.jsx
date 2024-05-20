import React, { useState, useContext } from "react";
import { Link } from "react-router-dom";
// import { SelectedCitiesContext } from './SelectCity';

function SelectPlace() {
  // const useSelectedCities = () => {
  //   return useContext(SelectedCitiesContext);
  // };

  // const { selectedCities, setSelectedCities } = useSelectedCities();

  const places = [
    {name: "인터스텔라", city: "서울"},
    {name: "레옹", city: "서울"},
    {name: "타이타닉", city: "런던"},
    {name: "어벤져스: 엔드게임", city: "서울"},
    {name: "쇼생크 탈출", city: "서울"},
    {name: "인셉션", city: "도쿄"},
    {name: "라라랜드", city: "서울"},
    {name: "캐스트 어웨이", city: "뉴욕"},
    {name: "겨울왕국", city: "서울"},
    {name: "미션 임파서블", city: "파리"}
  ]

  const [selectedCities, setSelectedCities] = useState('');
  const [selectedPlaces, setSelectedPlaces] = useState([]);

  const handleChange = (event) => {
    setSelectedCities(event.target.value);
  };

  const handlePlaceSelect = (placeName) => {
    setSelectedPlaces(prevSelectedPlaces => [...prevSelectedPlaces, placeName]);
  };

  const filteredPlaces = selectedCities ? places.filter(place => place.city === selectedCities).slice(0, 10) : places;

  return (
    <div>
      <h2>장소 도시 선택</h2>
      <select value={selectedCities} onChange={handleChange}>
        <option value="">도시를 선택하세요</option>
        <option value="서울">서울</option>
        <option value="뉴욕">뉴욕</option>
        <option value="런던">런던</option>
        <option value="도쿄">도쿄</option>
        <option value="파리">파리</option>
      </select>
      <div style={{ display: 'flex' }}>
        <div style={{ marginRight: '300px' }}>
          <h2>선택된 도시의 장소 목록</h2>
          <ul>
            {filteredPlaces.map(place => (
              <li key={place.name}>
                {place.name}
                <button onClick={() => handlePlaceSelect(place.name)}>선택</button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h2>선택된 장소 목록</h2>
          <ul>
            {selectedPlaces.map((place, index) => (
              <li key={index}>{place}</li>
            ))}
          </ul>
        </div>
      </div>
      <Link to="/">다음</Link>
    </div>
  );
};

export default SelectPlace;