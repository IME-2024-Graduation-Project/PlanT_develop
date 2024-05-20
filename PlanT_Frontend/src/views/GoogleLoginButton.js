import React from 'react';
import {
    Container,
    Row,
    Col,
} from "reactstrap";
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';
import axios from 'axios';
import { jwtDecode } from "jwt-decode";


const GoogleLoginButton = () => {

    const handleLoginSuccess = async (response) => {
        try {
            console.log(response);
            var profileObj = jwtDecode(response.credential);
            console.log(profileObj);
            const email = profileObj.email;
            const image = profileObj.picture;
            // 이메일을 Django로 보내어 사용자 생성 또는 확인
            const djangoResponse = await axios.post('http://localhost:8000/account/login/', { email });
            console.log(djangoResponse.data); // Django로부터 받은 사용자 정보 출력
            // 페이지를 다시 로드하여 이동
            localStorage.setItem("user", JSON.stringify(email));
            window.location.href = 'http://localhost:3000/admin/icons';
        } catch (error) {
            console.error(error);
        }
    };

    const handleLoginFailure = (error) => {
        console.error(error);
    };

    return (
        <Container>
            <Row className="justify-content-center">
                <Col xs="auto">
                    <GoogleOAuthProvider clientId="564324845555-4e3ovl5392l42f8fig3u8gc98vg7j8rm.apps.googleusercontent.com">
                        <GoogleLogin
                                onSuccess={handleLoginSuccess}
                                onFailure={handleLoginFailure}
                        />
                    </GoogleOAuthProvider>
                </Col>
            </Row>
        </Container>
    );
};

export default GoogleLoginButton;