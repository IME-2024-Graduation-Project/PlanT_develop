import React from "react";
import PlacesList from "variables/Places";
// reactstrap components

const MapMarker = ({ placeIds }) => {
  const mapRef = React.useRef(null);
  React.useEffect(() => {
    let google = window.google;
    let map = mapRef.current;
    let lat = "37.497917";
    let lng = "127.027628";
    const myLatlng = new google.maps.LatLng(lat, lng);
    const mapOptions = {
      zoom: 13,
      center: myLatlng,
      scrollwheel: false,
      zoomControl: true,
      styles: [
        {
          featureType: "water",
          stylers: [
            {
              saturation: 43,
            },
            {
              lightness: -11,
            },
            {
              hue: "#0088ff",
            },
          ],
        },
        {
          featureType: "road",
          elementType: "geometry.fill",
          stylers: [
            {
              hue: "#ff0000",
            },
            {
              saturation: -100,
            },
            {
              lightness: 99,
            },
          ],
        },
        {
          featureType: "road",
          elementType: "geometry.stroke",
          stylers: [
            {
              color: "#808080",
            },
            {
              lightness: 54,
            },
          ],
        },
        {
          featureType: "landscape.man_made",
          elementType: "geometry.fill",
          stylers: [
            {
              color: "#ece2d9",
            },
          ],
        },
        {
          featureType: "poi.park",
          elementType: "geometry.fill",
          stylers: [
            {
              color: "#ccdca1",
            },
          ],
        },
        {
          featureType: "road",
          elementType: "labels.text.fill",
          stylers: [
            {
              color: "#767676",
            },
          ],
        },
        {
          featureType: "road",
          elementType: "labels.text.stroke",
          stylers: [
            {
              color: "#ffffff",
            },
          ],
        },
        {
          featureType: "poi",
          stylers: [
            {
              visibility: "off",
            },
          ],
        },
        {
          featureType: "landscape.natural",
          elementType: "geometry.fill",
          stylers: [
            {
              visibility: "on",
            },
            {
              color: "#b8cb93",
            },
          ],
        },
        {
          featureType: "poi.park",
          stylers: [
            {
              visibility: "on",
            },
          ],
        },
        {
          featureType: "poi.sports_complex",
          stylers: [
            {
              visibility: "on",
            },
          ],
        },
        {
          featureType: "poi.medical",
          stylers: [
            {
              visibility: "on",
            },
          ],
        },
        {
          featureType: "poi.business",
          stylers: [
            {
              visibility: "simplified",
            },
          ],
        },
      ],
    };

    map = new google.maps.Map(mapRef.current, mapOptions);

     // 마커 정보를 저장할 배열
    const markers = [];


     // 모든 장소의 좌표를 이용하여 지도의 중심 위치와 마커를 생성합니다.
    const bounds = new google.maps.LatLngBounds();
    placeIds.forEach((placeId) => {
        const place = PlacesList.find((place) => place.id === placeId);

        // 만약 place가 존재한다면 마커를 생성합니다.
        if (place) {
          const position = new google.maps.LatLng(place.latitude, place.longitude);
    
          // LatLngBounds에 현재 좌표를 추가합니다.
          bounds.extend(position);
    
          // 마커 생성
          const marker = new google.maps.Marker({
            position,
            map,
            title: place.name,
          });
    
          // 마커를 markers 배열에 추가합니다.
          markers.push(marker);
 
       // 정보창 생성
       const contentString = `<div class="info-window-content"><h2>${marker.title}</h2>` +
                             `<p>Marker content here.</p></div>`;
       const infowindow = new google.maps.InfoWindow({
         content: contentString,
       });
 
       // 마커를 클릭했을 때 정보창 열기
       google.maps.event.addListener(marker, "click", function () {
         infowindow.open(map, marker);
       });
    }
        
    });

     // 모든 마커가 보이도록 지도 시점을 조절합니다.
    map.fitBounds(bounds);
    
   }, [placeIds]);
 
   return <div style={{ height: `100%` }} ref={mapRef}></div>;
};

export default MapMarker;
