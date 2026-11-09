import Foundation

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
