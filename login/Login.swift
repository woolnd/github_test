import Foundation

struct LoginManager {
    func isValidID(_ id: String) -> Bool {
        return id.count >= 4
    }

    func login(id: String, password: String) -> Bool {
        return isValidID(id) && !password.isEmpty
    }
}
