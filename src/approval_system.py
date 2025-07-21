import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"
    ESCALATED = "escalated"
    EXPIRED = "expired"

class ApprovalType(Enum):
    DOCUMENT_UPLOAD = "document_upload"
    POLICY_CHANGE = "policy_change"
    SENSITIVE_QUERY = "sensitive_query"
    BULK_OPERATION = "bulk_operation"

class EnterpriseApprovalWorkflow:
    """Enterprise-grade approval workflow system"""
    
    def __init__(self, data_dir: str = "approvals"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_approval_config()
        
        # Active requests storage
        self.requests_file = self.data_dir / "active_requests.json"
        self.history_file = self.data_dir / "approval_history.json"
        
        # Load existing requests
        self.active_requests = self._load_requests()
        self.approval_history = self._load_history()
        
    def _load_approval_config(self) -> Dict:
        """Load approval workflow configuration"""
        config = {
            'approver_roles': {
                'hr': {
                    'users': ['HR_MANAGER_USER_ID', 'HR_DIRECTOR_USER_ID'],
                    'required_approvals': 1,
                    'escalation_hours': 24
                },
                'technical': {
                    'users': ['TECH_LEAD_USER_ID', 'CTO_USER_ID'],
                    'required_approvals': 1,
                    'escalation_hours': 12
                },
                'policy': {
                    'users': ['POLICY_MANAGER_USER_ID', 'LEGAL_TEAM_USER_ID'],
                    'required_approvals': 2,
                    'escalation_hours': 48
                },
                'finance': {
                    'users': ['CFO_USER_ID', 'FINANCE_MANAGER_USER_ID'],
                    'required_approvals': 1,
                    'escalation_hours': 24
                },
                'security': {
                    'users': ['SECURITY_OFFICER_USER_ID', 'IT_SECURITY_USER_ID'],
                    'required_approvals': 1,
                    'escalation_hours': 8
                },
                'general': {
                    'users': ['ADMIN_USER_ID'],
                    'required_approvals': 1,
                    'escalation_hours': 72
                }
            },
            'approval_rules': {
                ApprovalType.DOCUMENT_UPLOAD.value: {
                    'auto_approve_extensions': ['.txt', '.md'],
                    'require_approval_extensions': ['.pdf', '.docx'],
                    'sensitive_keywords': ['confidential', 'secret', 'internal only', 'restricted'],
                    'max_auto_approve_size_mb': 1
                },
                ApprovalType.POLICY_CHANGE.value: {
                    'always_require_approval': True,
                    'minimum_approvers': 2
                },
                ApprovalType.SENSITIVE_QUERY.value: {
                    'sensitive_topics': ['salary', 'termination', 'legal', 'merger', 'acquisition'],
                    'require_approval': True
                }
            }
        }
        
        return config
    
    def _load_requests(self) -> Dict:
        """Load active approval requests"""
        if self.requests_file.exists():
            try:
                with open(self.requests_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load requests: {e}")
        return {}
    
    def _load_history(self) -> List:
        """Load approval history"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load history: {e}")
        return []
    
    def _save_requests(self):
        """Save active requests to disk"""
        try:
            with open(self.requests_file, 'w') as f:
                json.dump(self.active_requests, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save requests: {e}")
    
    def _save_history(self):
        """Save approval history to disk"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.approval_history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def create_approval_request(
        self, 
        requester_id: str, 
        approval_type: ApprovalType, 
        content: Dict[str, Any],
        category: str = 'general'
    ) -> str:
        """Create new approval request"""
        
        request_id = str(uuid.uuid4())
        
        # Determine if approval is needed
        needs_approval = self._needs_approval(approval_type, content, category)
        
        if not needs_approval:
            # Auto-approve
            return self._auto_approve_request(requester_id, approval_type, content)
        
        # Get approvers for category
        approver_config = self.config['approver_roles'].get(category, self.config['approver_roles']['general'])
        
        request = {
            'id': request_id,
            'requester_id': requester_id,
            'approval_type': approval_type.value,
            'category': category,
            'content': content,
            'status': ApprovalStatus.PENDING.value,
            'created_at': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(hours=approver_config['escalation_hours'])).isoformat(),
            'approvers': approver_config['users'],
            'required_approvals': approver_config['required_approvals'],
            'approver_responses': {},
            'escalation_level': 0
        }
        
        self.active_requests[request_id] = request
        self._save_requests()
        
        logger.info(f"Created approval request {request_id} for user {requester_id}")
        return request_id
    
    def _needs_approval(self, approval_type: ApprovalType, content: Dict, category: str) -> bool:
        """Determine if request needs approval"""
        
        rules = self.config['approval_rules'].get(approval_type.value, {})
        
        if approval_type == ApprovalType.DOCUMENT_UPLOAD:
            file_name = content.get('file_name', '')
            file_size_mb = content.get('file_size_mb', 0)
            file_content = content.get('preview', '').lower()
            file_ext = Path(file_name).suffix.lower()
            
            # Check if auto-approve conditions are met
            if (file_ext in rules.get('auto_approve_extensions', []) and 
                file_size_mb <= rules.get('max_auto_approve_size_mb', 0) and
                not any(keyword in file_content for keyword in rules.get('sensitive_keywords', []))):
                return False
            
            return True
            
        elif approval_type == ApprovalType.SENSITIVE_QUERY:
            query_text = content.get('query', '').lower()
            sensitive_topics = rules.get('sensitive_topics', [])
            
            return any(topic in query_text for topic in sensitive_topics)
        
        # Default to requiring approval for unknown types
        return True
    
    def _auto_approve_request(self, requester_id: str, approval_type: ApprovalType, content: Dict) -> str:
        """Auto-approve request without human intervention"""
        
        request_id = f"auto_{str(uuid.uuid4())[:8]}"
        
        auto_approval_record = {
            'id': request_id,
            'requester_id': requester_id,
            'approval_type': approval_type.value,
            'content': content,
            'status': ApprovalStatus.APPROVED.value,
            'created_at': datetime.now().isoformat(),
            'approved_at': datetime.now().isoformat(),
            'approval_method': 'automatic',
            'approver_id': 'system'
        }
        
        self.approval_history.append(auto_approval_record)
        self._save_history()
        
        return request_id
    
    def get_approval_blocks(self, request_id: str) -> List[Dict]:
        """Generate Slack blocks for approval request"""
        
        request = self.active_requests.get(request_id)
        if not request:
            return []
        
        content = request['content']
        approval_type = request['approval_type']
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🔐 Approval Required: {approval_type.replace('_', ' ').title()}"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Request ID:* {request_id[:8]}..."
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Submitted by:* <@{request['requester_id']}>"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Category:* {request['category']}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:* {request['status']}"
                    }
                ]
            }
        ]
        
        # Add content-specific information
        if approval_type == 'document_upload':
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*📁 File Details:*\n• Name: `{content.get('file_name', 'Unknown')}`\n• Size: {content.get('file_size_mb', 0):.1f} MB\n• Type: {content.get('file_type', 'Unknown')}"
                }
            })
            
            if content.get('preview'):
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*📄 Content Preview:*\n``````"
                    }
                })
        
        # Add approval actions
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "✅ Approve"
                    },
                    "style": "primary",
                    "value": f"approve_{request_id}",
                    "action_id": "approve_request"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "❌ Reject"
                    },
                    "style": "danger",
                    "value": f"reject_{request_id}",
                    "action_id": "reject_request"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "🔄 Request Changes"
                    },
                    "value": f"changes_{request_id}",
                    "action_id": "request_changes"
                }
            ]
        })
        
        # Add expiration warning
        expires_at = datetime.fromisoformat(request['expires_at'])
        time_left = expires_at - datetime.now()
        
        if time_left.total_seconds() > 0:
            hours_left = int(time_left.total_seconds() / 3600)
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"⏰ Expires in {hours_left} hours | Required approvals: {request['required_approvals']}"
                    }
                ]
            })
        else:
            blocks.append({
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": "⚠️ This request has expired and will be escalated"
                    }
                ]
            })
        
        return blocks
    
    def process_approval_response(
        self, 
        request_id: str, 
        approver_id: str, 
        action: str, 
        comments: str = ""
    ) -> Dict[str, Any]:
        """Process approval response from approver"""
        
        if request_id not in self.active_requests:
            return {'success': False, 'message': 'Request not found or already processed'}
        
        request = self.active_requests[request_id]
        
        # Check if user is authorized approver
        if approver_id not in request['approvers']:
            return {'success': False, 'message': 'You are not authorized to approve this request'}
        
        # Check if request has expired
        expires_at = datetime.fromisoformat(request['expires_at'])
        if datetime.now() > expires_at:
            return {'success': False, 'message': 'This request has expired'}
        
        # Record approver response
        request['approver_responses'][approver_id] = {
            'action': action,
            'comments': comments,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate current status
        approvals = sum(1 for resp in request['approver_responses'].values() if resp['action'] == 'approve')
        rejections = sum(1 for resp in request['approver_responses'].values() if resp['action'] == 'reject')
        changes_requested = sum(1 for resp in request['approver_responses'].values() if resp['action'] == 'changes')
        
        # Determine final status
        if rejections > 0:
            request['status'] = ApprovalStatus.REJECTED.value
            self._move_to_history(request)
        elif changes_requested > 0:
            request['status'] = ApprovalStatus.CHANGES_REQUESTED.value
        elif approvals >= request['required_approvals']:
            request['status'] = ApprovalStatus.APPROVED.value
            self._move_to_history(request)
        
        self._save_requests()
        
        return {
            'success': True, 
            'status': request['status'], 
            'request': request,
            'approvals_needed': max(0, request['required_approvals'] - approvals)
        }
    
    def _move_to_history(self, request: Dict):
        """Move completed request to history"""
        
        request['completed_at'] = datetime.now().isoformat()
        self.approval_history.append(request.copy())
        
        # Remove from active requests
        if request['id'] in self.active_requests:
            del self.active_requests[request['id']]
        
        self._save_history()
        self._save_requests()
    
    def check_expired_requests(self):
        """Check for expired requests and handle escalation"""
        
        current_time = datetime.now()
        expired_requests = []
        
        for request_id, request in self.active_requests.items():
            expires_at = datetime.fromisoformat(request['expires_at'])
            
            if current_time > expires_at and request['status'] == ApprovalStatus.PENDING.value:
                expired_requests.append(request_id)
        
        # Handle expired requests
        for request_id in expired_requests:
            self._escalate_request(request_id)
        
        return len(expired_requests)
    
    def _escalate_request(self, request_id: str):
        """Escalate expired request"""
        
        request = self.active_requests[request_id]
        request['escalation_level'] += 1
        request['status'] = ApprovalStatus.ESCALATED.value
        
        # Extend expiration time
        new_expiration = datetime.now() + timedelta(hours=24)
        request['expires_at'] = new_expiration.isoformat()
        
        # Add escalation approvers (e.g., higher management)
        escalation_approvers = ['ESCALATION_MANAGER_ID', 'DIRECTOR_ID']
        request['approvers'].extend(escalation_approvers)
        
        self._save_requests()
        
        logger.warning(f"Escalated request {request_id} to level {request['escalation_level']}")
    
    def get_pending_approvals_for_user(self, user_id: str) -> List[Dict]:
        """Get pending approval requests for a specific approver"""
        
        pending = []
        for request in self.active_requests.values():
            if (user_id in request['approvers'] and 
                request['status'] == ApprovalStatus.PENDING.value and
                user_id not in request['approver_responses']):
                pending.append(request)
        
        return pending
    
    def get_approval_statistics(self) -> Dict[str, Any]:
        """Get approval workflow statistics"""
        
        # Count active requests by status
        status_counts = {}
        for request in self.active_requests.values():
            status = request['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Count historical approvals
        total_processed = len(self.approval_history)
        approved_count = sum(1 for req in self.approval_history if req['status'] == ApprovalStatus.APPROVED.value)
        rejected_count = sum(1 for req in self.approval_history if req['status'] == ApprovalStatus.REJECTED.value)
        
        # Calculate average processing time
        processing_times = []
        for req in self.approval_history:
            if req.get('completed_at'):
                created = datetime.fromisoformat(req['created_at'])
                completed = datetime.fromisoformat(req['completed_at'])
                processing_times.append((completed - created).total_seconds() / 3600)  # Hours
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            'active_requests': status_counts,
            'total_processed': total_processed,
            'approval_rate': (approved_count / max(1, total_processed)) * 100,
            'rejection_rate': (rejected_count / max(1, total_processed)) * 100,
            'average_processing_hours': avg_processing_time,
            'pending_count': len([r for r in self.active_requests.values() if r['status'] == ApprovalStatus.PENDING.value])
        }
