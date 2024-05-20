import React, { createContext, useContext, useState } from 'react';

// UserContext를 생성합니다.
const UserContext = createContext();

// UserContext의 Provider를 사용하기 위한 커스텀 훅을 만듭니다.
const useUser = () => useContext(UserContext);

// UserProvider 컴포넌트를 생성합니다.
const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);

  return (
    <UserContext.Provider value={{ user, setUser }}>
      {children}
    </UserContext.Provider>
  );
};


export { UserContext, UserProvider };