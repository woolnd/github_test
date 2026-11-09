import Foundation

enum LoginError: Error {
    case invalidID
    case invalidPassword

    var message: String {
        switch self {
        case .invalidID: return "아이디는 4자 이상이어야 합니다."
        case .invalidPassword: return "비밀번호는 8자 이상이어야 합니다."
        }
    }
}

struct LoginManager {
    func isValidID(_ id: String) -> Bool {
        return id.count >= 4
    }

    func isValidPassword(_ password: String) -> Bool {
        return password.count >= 8
    }

    func login(id: String, password: String) -> Bool {
        return isValidID(id) && isValidPassword(password)
    }
}
