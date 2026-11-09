import Foundation

struct LoginManager {
    func login(id: String, password: String) -> Bool {
        return !id.isEmpty && !password.isEmpty
    }
}
